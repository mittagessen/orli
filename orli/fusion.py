# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#           2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Llama vision fusion model"""

import logging
from typing import Optional

import json
import math
import torch
from torch import nn
import torch.nn.functional as F

from orli.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                          TransformerCrossAttentionLayer, FeedForward,
                          TransformerDecoder, TransformerSelfAttentionLayer,
                          FusionLayer, scale_hidden_dim_for_mlp,
                          Llama3ScaledRoPE, llama3_mlp,
                          PositionEmbeddingRandom)
from orli.modules.transformer import _get_clones
from orli.modules.baseline import (DEFAULT_NUM_BASELINE_POINTS,
                                   baseline_param_dim,
                                   prepare_baseline_anchors)

logger = logging.getLogger(__name__)

__all__ = ['baseline_decoder', 'OrliAdapter', 'OrliHybridNeck', 'CurveRegressionHead']


class CurveTokenEmbedding(nn.Module):
    """
    Separate embeddings for token classes and curve coordinates, with optional
    Fourier features for the curve input.
    """
    def __init__(self,
                 token_dim: int,
                 curve_dim: int,
                 embed_dim: int,
                 num_curve_freqs: int = 4):
        super().__init__()
        self.token_dim = token_dim
        self.curve_dim = curve_dim
        self.embed_dim = embed_dim
        self.num_curve_freqs = num_curve_freqs

        curve_in_dim = curve_dim * (1 + 2 * num_curve_freqs) if num_curve_freqs > 0 else curve_dim
        self.tok_proj = nn.Linear(token_dim, embed_dim, bias=False)
        self.curve_proj = nn.Linear(curve_in_dim, embed_dim, bias=False)

        if num_curve_freqs > 0:
            freqs = 2 ** torch.arange(num_curve_freqs, dtype=torch.float32)
            self.register_buffer('curve_freqs', freqs, persistent=False)
        else:
            self.curve_freqs = None

    def _curve_features(self, curves: torch.Tensor) -> torch.Tensor:
        if self.num_curve_freqs <= 0:
            return curves
        freqs = self.curve_freqs.to(dtype=curves.dtype, device=curves.device)
        # [b, s, curve_dim, num_freqs]
        scaled = curves.unsqueeze(-1) * (2.0 * math.pi * freqs)
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        sin = sin.flatten(start_dim=-2)
        cos = cos.flatten(start_dim=-2)
        return torch.cat([curves, sin, cos], dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: [b, s, token_dim + curve_dim]
        tok = tokens[..., :self.token_dim]
        curves = tokens[..., self.token_dim:self.token_dim + self.curve_dim]
        curve_feat = self._curve_features(curves)
        return self.tok_proj(tok) + self.curve_proj(curve_feat)


def baseline_decoder(vocab_size: int = 4 + baseline_param_dim(DEFAULT_NUM_BASELINE_POINTS),
                     num_layers: int = 12,
                     num_heads: int = 9,
                     num_kv_heads: int = 3,
                     embed_dim: int = 576,
                     max_seq_len: int = 768,
                     intermediate_dim: int = 1536,
                     attn_dropout: int = 0.0,
                     norm_eps: int = 1e-5,
                     rope_base: int = 10000,
                     encoder_sizes: list[tuple[int, int]] = None,  # start of fusion parameters
                     curve_num_freqs: int = 4,
                     pretrained: Optional[str] = None,
                     **kwargs) -> TransformerDecoder:
    """
    Builds a decoder regressing ordered local-frame baseline vectors using
    iterative refinement.

    Args:
        vocab_size (int): dimensionality of baseline encoding
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value.
        num_kv_heads (int): number of key and value heads. User should ensure
            `num_heads` % `num_kv_heads` == 0. For standard MHA set `num_kv_heads` == `num_heads`,
            for GQA `num_kv_heads` < `num_heads`, and for MQA set `num_kv_heads` == 1.
        embed_dim (int): embedding dimension for self-attention.
        max_seq_len (int): maximum sequence length the model will be run with, as used
            by :func:`~party.modules.KVCache`.
        intermediate_dim (Optional[int]): intermediate dimension for MLP. If not specified,
            this is computed using :func:`~party.modules.scale_hidden_dim_for_mlp`.
        encoder_max_seq_len (int): maximum sequence length the encoder will be run with, as used
            by :func:`~party.modules.KVCache`.
        pretrained (str): huggingface hub identifier of pretrained bytellama
                          weights. All hyperparameters will except
                          encoder_max_seq_len will be ignored.

    Returns:
        TransformerDecoder: Instantiation of Llama 3.2 vision decoder.
    """
    encoder_max_seq_len = sum([x[0] * x[1] for x in encoder_sizes])

    config = {'vocab_size': vocab_size,
              'num_layers': num_layers,
              'num_heads': num_heads,
              'num_kv_heads': num_kv_heads,
              'embed_dim': embed_dim,
              'max_seq_len': max_seq_len,
              'intermediate_dim': intermediate_dim,
              'attn_dropout': attn_dropout,
              'norm_eps': norm_eps,
              'rope_base': rope_base,
              'encoder_max_seq_len': encoder_max_seq_len}

    if pretrained:
        vocab_size = config.pop('vocab_size')
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo_id=pretrained, filename='config.json'), 'r') as fp:
            config.update(json.load(fp))
        config['vocab_size'] = vocab_size

    head_dim = config['embed_dim'] // config['num_heads']
    num_kv_heads = config['num_kv_heads'] if config['num_kv_heads'] else config['num_heads']
    hidden_dim = config['intermediate_dim'] or scale_hidden_dim_for_mlp(config['embed_dim'])
    layers = []

    rope = Llama3ScaledRoPE(dim=head_dim, max_seq_len=config['max_seq_len'], base=config['rope_base'])

    for idx in range(1, num_layers + 1):

        # Self attention layers for text decoder
        self_attn = MultiHeadAttention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_kv_heads=config['num_kv_heads'],
            head_dim=head_dim,
            q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
            k_proj=nn.Linear(config['embed_dim'], config['num_kv_heads'] * head_dim, bias=False),
            v_proj=nn.Linear(config['embed_dim'], config['num_kv_heads'] * head_dim, bias=False),
            output_proj=nn.Linear(config['embed_dim'], config['embed_dim'], bias=False),
            pos_embeddings=rope,
            max_seq_len=config['max_seq_len'],
            attn_dropout=0.0,
        )
        mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
        decoder_layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=RMSNorm(dim=embed_dim, eps=1e-5),
            mlp_norm=RMSNorm(dim=embed_dim, eps=1e-5),
        )

        attn = MultiHeadAttention(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_kv_heads=config['num_kv_heads'],
            head_dim=head_dim,
            q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
            k_proj=nn.Linear(config['embed_dim'], config['num_kv_heads'] * head_dim, bias=False),
            v_proj=nn.Linear(config['embed_dim'], config['num_kv_heads'] * head_dim, bias=False),
            output_proj=nn.Linear(config['embed_dim'], config['embed_dim'], bias=False),
            q_norm=RMSNorm(dim=head_dim, eps=1e-05),
            k_norm=RMSNorm(dim=head_dim, eps=1e-05),
            pos_embeddings=None,
            max_seq_len=config['encoder_max_seq_len'],
            is_causal=False,
            attn_dropout=0.0,
        )

        xattn_mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
        xattn_layer = TransformerCrossAttentionLayer(
            attn=attn,
            mlp=xattn_mlp,
            ca_norm=RMSNorm(dim=embed_dim),
            mlp_norm=RMSNorm(dim=embed_dim),
            ca_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )
        layers.append(FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer))

    token_dim = 4
    curve_dim = config['vocab_size'] - token_dim
    if curve_dim <= 0:
        raise ValueError(f'vocab_size ({config["vocab_size"]}) must be > token_dim ({token_dim}).')
    line_embeddings = CurveTokenEmbedding(token_dim=token_dim,
                                          curve_dim=curve_dim,
                                          embed_dim=config['embed_dim'],
                                          num_curve_freqs=curve_num_freqs)

    decoder = TransformerDecoder(tok_embeddings=line_embeddings,
                                 layers=layers,
                                 max_seq_len=config['max_seq_len'],
                                 num_heads=config['num_heads'],
                                 head_dim=head_dim,
                                 norm=RMSNorm(config['embed_dim'], eps=1e-05),
                                 output=nn.Identity(),
                                 output_hidden_states=[2, 5, 8])

    if pretrained:
        weight_path = hf_hub_download(repo_id=pretrained, filename='model.safetensors')
        from safetensors import safe_open
        with safe_open(weight_path, framework='pt') as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def build_2d_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == 'batch':
        return nn.BatchNorm2d(channels)
    if norm_type == 'group':
        return nn.GroupNorm(_group_count(channels), channels)
    if norm_type == 'layer':
        return LayerNorm2d(channels)
    raise ValueError(f'Unsupported neck normalization: {norm_type}')


def build_activation(name: str) -> nn.Module:
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'silu':
        return nn.SiLU(inplace=True)
    raise ValueError(f'Unsupported activation: {name}')


def build_2d_sincos_position_embedding(height: int,
                                       width: int,
                                       embed_dim: int,
                                       temperature: float = 10000.0,
                                       *,
                                       device: Optional[torch.device] = None,
                                       dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError('Embedding dimension must be divisible by 4 for 2D sin-cos PE.')
    grid_y = torch.arange(height, dtype=torch.float32, device=device)
    grid_x = torch.arange(width, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    pe = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)
    return pe.to(dtype=dtype).unsqueeze(0)


class ConvNormAct2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 *,
                 groups: int = 1,
                 padding: Optional[int] = None,
                 norm_type: str = 'group',
                 act: Optional[str] = 'silu'):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.norm = build_2d_norm(norm_type, out_channels)
        self.act = nn.Identity() if act is None else build_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 norm_type: str = 'group',
                 act: str = 'silu'):
        super().__init__()
        self.conv1 = ConvNormAct2d(in_channels, out_channels, 3, norm_type=norm_type, act=act)
        self.conv2 = ConvNormAct2d(out_channels, out_channels, 3, norm_type=norm_type, act=None)
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         ConvNormAct2d(in_channels, out_channels, 1, norm_type=norm_type, act=None))
        self.act = build_activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class FusionConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 *,
                 depth: int = 2,
                 norm_type: str = 'group',
                 act: str = 'silu'):
        super().__init__()
        layers = [ResidualConvBlock(in_channels, out_channels, norm_type=norm_type, act=act)]
        for _ in range(depth - 1):
            layers.append(ResidualConvBlock(out_channels, out_channels, norm_type=norm_type, act=act))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.0,
                 act: str = 'gelu'):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, dim_feedforward),
                                 build_activation(act),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.norm1(x)
        if pos_embed is not None:
            y = y + pos_embed
        attn_out, _ = self.self_attn(y, y, value=self.norm1(x), need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SpatialTransformerEncoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dim_feedforward: int,
                 num_layers: int,
                 dropout: float = 0.0,
                 act: str = 'gelu'):
        super().__init__()
        self.layers = nn.ModuleList([SpatialTransformerEncoderLayer(embed_dim,
                                                                    num_heads,
                                                                    dim_feedforward,
                                                                    dropout=dropout,
                                                                    act=act)
                                     for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, pos_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed)
        return x


class OrliAdapter(nn.Module):
    """
    Builds an adapter head consisting of depthwise-separable downsampling
    convolutions, `num_layers` self attention layers, a linear projection of
    encoder_embed_dim to decoder_embed_dim, and 2D positional embeddings.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 ds_factors: list[int] = None):
        super().__init__()
        if ds_factors is None:
            ds_factors = [2] * len(encoder_embed_dims)
        mlp_ratio = 4
        num_kv_heads = num_heads
        self.adapter = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.pos_embeddings = nn.ModuleList()
        self.output_sizes: list[tuple[int, int]] = []

        for encoder_embed_dim, size, ds_factor in zip(encoder_embed_dims, encoder_sizes, ds_factors):
            hidden_dim = int(mlp_ratio * encoder_embed_dim)
            head_dim = encoder_embed_dim // num_heads

            # depthwise-separable downsampling convolution
            if ds_factor > 1:
                self.downsample.append(nn.Sequential(
                    nn.Conv2d(encoder_embed_dim, encoder_embed_dim,
                              kernel_size=ds_factor, stride=ds_factor,
                              groups=encoder_embed_dim, bias=False),
                    nn.Conv2d(encoder_embed_dim, encoder_embed_dim,
                              kernel_size=1, bias=False),
                ))
            else:
                self.downsample.append(nn.Identity())
            ds_size = (size[0] // ds_factor, size[1] // ds_factor)
            self.output_sizes.append(ds_size)

            layers = []
            for _ in range(num_layers):
                self_attn = MultiHeadAttention(embed_dim=encoder_embed_dim,
                                               num_heads=num_heads,
                                               num_kv_heads=num_heads,
                                               head_dim=head_dim,
                                               q_proj=nn.Linear(encoder_embed_dim, num_heads * head_dim, bias=False),
                                               k_proj=nn.Linear(encoder_embed_dim, num_kv_heads * head_dim, bias=False),
                                               v_proj=nn.Linear(encoder_embed_dim, num_kv_heads * head_dim, bias=False),
                                               output_proj=nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=False),
                                               pos_embeddings=None,
                                               attn_dropout=0.0,
                                               is_causal=False)

                mlp = FeedForward(gate_proj=nn.Linear(encoder_embed_dim, hidden_dim),
                                  down_proj=nn.Linear(hidden_dim, encoder_embed_dim),
                                  up_proj=None)

                layer = TransformerSelfAttentionLayer(attn=self_attn,
                                                      mlp=mlp,
                                                      sa_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                      mlp_norm=RMSNorm(encoder_embed_dim, eps=1e-5),
                                                      sa_scale=TanhGate(),
                                                      mlp_scale=TanhGate())
                layers.append(layer)
            layers.append(nn.Linear(encoder_embed_dim, decoder_embed_dim))
            self.adapter.append(nn.Sequential(*layers))
            self.pos_embeddings.append(PositionEmbeddingRandom(decoder_embed_dim, ds_size))

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        os = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            hidden_state = self.downsample[idx](hidden_state)
            hidden_state = hidden_state.flatten(-2).transpose(-1, -2)
            hidden_state = self.adapter[idx](hidden_state)
            os.append(self.pos_embeddings[idx](hidden_state))
        return torch.cat(os, dim=1)


class OrliHybridNeck(nn.Module):
    """
    A hybrid multiscale neck with shared channel projection, optional
    transformer encoding on selected scales, top-down FPN fusion, and
    bottom-up PAN fusion before flattening into decoder memory tokens.
    """

    def __init__(self,
                 encoder_embed_dims: list[int],
                 encoder_sizes: list[tuple[int, int]],
                 decoder_embed_dim: int,
                 hidden_dim: int = 256,
                 num_heads: int = 8,
                 num_encoder_layers: int = 1,
                 use_encoder_idx: Optional[list[int]] = None,
                 output_ds_factors: Optional[list[int]] = None,
                 norm_type: str = 'group',
                 dim_feedforward: int = 1024,
                 dropout: float = 0.0,
                 fusion_depth: int = 2,
                 transformer_act: str = 'gelu',
                 conv_act: str = 'silu'):
        super().__init__()
        if output_ds_factors is None:
            output_ds_factors = [2] * len(encoder_embed_dims)
        if len(output_ds_factors) != len(encoder_embed_dims):
            raise ValueError('output_ds_factors must match the number of encoder feature maps.')
        if use_encoder_idx is None:
            use_encoder_idx = [len(encoder_embed_dims) - 1]
        use_encoder_idx = sorted(use_encoder_idx)
        if any(idx < 0 or idx >= len(encoder_embed_dims) for idx in use_encoder_idx):
            raise ValueError('use_encoder_idx contains invalid feature map indices.')

        self.hidden_dim = hidden_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.encoder_sizes = encoder_sizes
        self.output_ds_factors = list(output_ds_factors)
        self.use_encoder_idx = use_encoder_idx
        self.output_sizes = [(size[0] // ds_factor, size[1] // ds_factor)
                             for size, ds_factor in zip(encoder_sizes, self.output_ds_factors)]

        self.input_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(channels,
                                                                 hidden_dim,
                                                                 kernel_size=1,
                                                                 bias=False),
                                                       build_2d_norm(norm_type, hidden_dim))
                                         for channels in encoder_embed_dims])

        self.encoders = nn.ModuleList([SpatialTransformerEncoder(hidden_dim,
                                                                 num_heads,
                                                                 dim_feedforward,
                                                                 num_encoder_layers,
                                                                 dropout=dropout,
                                                                 act=transformer_act)
                                       for _ in self.use_encoder_idx])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(encoder_embed_dims) - 1):
            self.lateral_convs.append(ConvNormAct2d(hidden_dim, hidden_dim, 1, norm_type=norm_type, act=conv_act))
            self.fpn_blocks.append(FusionConvBlock(hidden_dim * 2,
                                                   hidden_dim,
                                                   depth=fusion_depth,
                                                   norm_type=norm_type,
                                                   act=conv_act))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(encoder_embed_dims) - 1):
            self.downsample_convs.append(ConvNormAct2d(hidden_dim,
                                                       hidden_dim,
                                                       3,
                                                       stride=2,
                                                       norm_type=norm_type,
                                                       act=conv_act))
            self.pan_blocks.append(FusionConvBlock(hidden_dim * 2,
                                                   hidden_dim,
                                                   depth=fusion_depth,
                                                   norm_type=norm_type,
                                                   act=conv_act))

        self.output_downsample = nn.ModuleList([self._make_output_downsample(hidden_dim, ds_factor, norm_type, conv_act)
                                                for ds_factor in self.output_ds_factors])
        self.output_proj = nn.ModuleList([nn.Conv2d(hidden_dim, decoder_embed_dim, kernel_size=1, bias=False)
                                          for _ in encoder_embed_dims])
        self.level_embeddings = nn.Parameter(torch.zeros(len(encoder_embed_dims), decoder_embed_dim))
        nn.init.normal_(self.level_embeddings, std=0.02)

    def _make_output_downsample(self,
                                channels: int,
                                ds_factor: int,
                                norm_type: str,
                                act: str) -> nn.Module:
        if ds_factor <= 1:
            return nn.Identity()
        return nn.Sequential(
            ConvNormAct2d(channels,
                          channels,
                          ds_factor,
                          stride=ds_factor,
                          groups=channels,
                          padding=0,
                          norm_type=norm_type,
                          act=act),
            ConvNormAct2d(channels, channels, 1, norm_type=norm_type, act=act),
        )

    def _encode_scale(self, feat: torch.Tensor, encoder: SpatialTransformerEncoder) -> torch.Tensor:
        batch_size, _, height, width = feat.shape
        src = feat.flatten(2).transpose(1, 2)
        pos_embed = build_2d_sincos_position_embedding(height,
                                                       width,
                                                       self.hidden_dim,
                                                       device=src.device,
                                                       dtype=src.dtype)
        src = encoder(src, pos_embed=pos_embed)
        return src.transpose(1, 2).reshape(batch_size, self.hidden_dim, height, width).contiguous()

    def _flatten_output(self, feat: torch.Tensor, level_idx: int) -> torch.Tensor:
        feat = self.output_downsample[level_idx](feat)
        feat = self.output_proj[level_idx](feat)
        _, _, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        pos_embed = build_2d_sincos_position_embedding(height,
                                                       width,
                                                       self.decoder_embed_dim,
                                                       device=tokens.device,
                                                       dtype=tokens.dtype)
        level_embed = self.level_embeddings[level_idx].view(1, 1, -1)
        return tokens + pos_embed + level_embed

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        proj_feats = [proj(feat) for proj, feat in zip(self.input_proj, encoder_hidden_states)]

        for encoder, feat_idx in zip(self.encoders, self.use_encoder_idx):
            proj_feats[feat_idx] = self._encode_scale(proj_feats[feat_idx], encoder)

        inner_outs = [proj_feats[-1]]
        for level_idx, feat_idx in enumerate(range(len(proj_feats) - 1, 0, -1)):
            high = self.lateral_convs[level_idx](inner_outs[0])
            low = proj_feats[feat_idx - 1]
            inner_outs[0] = high
            upsampled = F.interpolate(high, size=low.shape[-2:], mode='nearest')
            inner_out = self.fpn_blocks[level_idx](torch.cat([upsampled, low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for level_idx in range(len(inner_outs) - 1):
            low = outs[-1]
            high = inner_outs[level_idx + 1]
            downsampled = self.downsample_convs[level_idx](low)
            outs.append(self.pan_blocks[level_idx](torch.cat([downsampled, high], dim=1)))

        return torch.cat([self._flatten_output(feat, level_idx) for level_idx, feat in enumerate(outs)], dim=1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Inverse sigmoid (logit) function with clamping for numerical stability.
    """
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


class CurveRegressionHead(nn.Module):
    """
    Iterative refinement regression head for local-frame baseline vectors.

    Uses sigmoid/inverse-sigmoid refinement: offsets are predicted in logit
    space and combined with the previous iteration's baseline vector via inverse
    sigmoid, addition, and sigmoid. This ensures outputs stay in [0,1] and
    provides better gradient flow near boundaries. Each refinement step
    conditions its offset prediction on the decoder state, the selected anchor
    identity, and the current baseline state.
    """

    def __init__(self,
                 anchors: tuple[tuple[float, ...], ...],
                 embed_dim: int = 576,
                 num_layers: int = 3,
                 num_iterations: int = 4,
                 num_baseline_points: int = DEFAULT_NUM_BASELINE_POINTS,
                 direct_point_regression: bool = False,
                 anchor_embedding: bool = True):
        super().__init__()
        if isinstance(anchors, torch.Tensor):
            anchors_t = anchors.float()
        else:
            anchors_t = torch.tensor(anchors, dtype=torch.float32)
        anchors_t = prepare_baseline_anchors(anchors_t,
                                             num_points=num_baseline_points,
                                             direct_point_regression=direct_point_regression)
        self.num_anchors = anchors_t.shape[0]
        self.num_baseline_points = int(num_baseline_points)
        self.direct_point_regression = bool(direct_point_regression)
        self.use_anchor_embedding = bool(anchor_embedding)
        self.curve_dim = anchors_t.shape[-1]
        num_cls = 4
        reg_hidden_dim = scale_hidden_dim_for_mlp(embed_dim)

        reg_proj = nn.Sequential()
        self.register_buffer('curve_anchors', anchors_t.clamp(1e-5, 1 - 1e-5))

        self.norms = nn.ModuleList([RMSNorm(embed_dim) for _ in range(num_iterations)])

        if num_layers > 1:
            for n in range(num_layers-1):
                reg_proj.append(nn.Linear(reg_hidden_dim if n else embed_dim, reg_hidden_dim))
                reg_proj.append(nn.SiLU())
        reg_proj.append(nn.Linear(reg_hidden_dim if num_layers > 1 else embed_dim, self.curve_dim))
        # zero-initialize the final layer for near-zero initial offsets
        nn.init.zeros_(reg_proj[-1].weight)
        nn.init.zeros_(reg_proj[-1].bias)
        cls_proj = nn.Linear(embed_dim, num_cls)
        reg_input_proj = nn.Linear(embed_dim * 3, embed_dim)
        nn.init.zeros_(reg_input_proj.bias)
        with torch.no_grad():
            reg_input_proj.weight.zero_()
            reg_input_proj.weight[:, :embed_dim].copy_(torch.eye(embed_dim,
                                                                 dtype=reg_input_proj.weight.dtype))
            reg_input_proj.weight[:, embed_dim:].normal_(std=1e-3)
        self.anchor_proj = nn.Linear(embed_dim, self.num_anchors)
        if self.use_anchor_embedding:
            self.anchor_embeddings = nn.Embedding(self.num_anchors, embed_dim)
        else:
            self.anchor_embeddings = None
        self.curve_state_proj = nn.Linear(self.curve_dim, embed_dim, bias=False)
        self.reg_input_projs = _get_clones(reg_input_proj, num_iterations)
        self.reg_projs = _get_clones(reg_proj, num_iterations)
        self.cls_projs = _get_clones(cls_proj, num_iterations)

    def forward(self,
                xs: list[torch.Tensor],
                target_anchor_idx: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        """
        Args:
            xs: A list containing `num_iterations` tensors of shape
                ``[b, s, d_e]`` where ``d_e`` is the decoder embedding dim.
            target_anchor_idx: Optional tensor of shape ``[b, s]`` with
                per-token anchor ids used for training-time anchor
                initialization.

        Returns:
            A dictionary containing an output tensor with shape
            ``[num_iterations, b, s, curve_dim]`` under the key `curves` and the
            class logits in `tokens` with shape ``[num_iterations, b, s, 4]``.
            The first-step anchor logits are returned under `anchors` with shape
            ``[b, s, num_anchors]``.
        """
        # anchor selection: use a dedicated first-step anchor head to pick the
        # per-token initialization anchor without overloading the token logits.
        h0 = self.norms[0](xs[0])
        anchor_logits = self.anchor_proj(h0)  # [b, s, N]
        if target_anchor_idx is not None:
            anchor_idx = target_anchor_idx.clamp(min=0, max=self.num_anchors - 1)
        else:
            anchor_idx = anchor_logits.argmax(-1)  # [b, s]
        init_curves = self.curve_anchors[anchor_idx]  # [b, s, curve_dim]
        if self.use_anchor_embedding:
            anchor_cond = self.anchor_embeddings(anchor_idx)
        else:
            anchor_cond = torch.zeros_like(h0)

        _curves: list[torch.Tensor] = [init_curves]
        _logits: list[torch.Tensor] = []
        for norm, cls_proj, reg_input_proj, reg_proj, layer in zip(self.norms,
                                                                   self.cls_projs,
                                                                   self.reg_input_projs,
                                                                   self.reg_projs,
                                                                   xs):
            curves = _curves[-1]
            layer = norm(layer)
            curve_state = inverse_sigmoid(curves)
            reg_input = torch.cat([layer,
                                   anchor_cond,
                                   self.curve_state_proj(curve_state)],
                                  dim=-1)
            offsets = reg_proj(reg_input_proj(reg_input))
            _curves.append(torch.sigmoid(inverse_sigmoid(curves) + offsets))
            _logits.append(cls_proj(layer))

        return {'curves': torch.stack(_curves[1:]),
                'tokens': torch.stack(_logits),
                'anchors': anchor_logits}
