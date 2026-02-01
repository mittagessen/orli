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
from typing import Optional, Union, TYPE_CHECKING

import json
import torch
from torch import nn

from orli.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                          TransformerCrossAttentionLayer, FeedForward,
                          TransformerDecoder, TransformerSelfAttentionLayer,
                          FusionLayer, scale_hidden_dim_for_mlp,
                          Llama3ScaledRoPE, llama3_mlp)
from orli.modules.transformer import _get_clones

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)

__all__ = ['baseline_decoder', 'CurveRegressionHead']


def baseline_decoder(vocab_size: int = 12,
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
                     fusion_interval: int = 2,
                     pretrained: Optional[str] = None,
                     **kwargs) -> TransformerDecoder:
    """
    Builds a decoder regressing baselines as cubic Bézier curves using
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
              'encoder_max_seq_len': encoder_max_seq_len,
              'fusion_interval': fusion_interval}

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

        if idx % config['fusion_interval'] == 0:
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

            mlp = llama3_mlp(dim=config['embed_dim'], hidden_dim=hidden_dim)
            xattn_layer = TransformerCrossAttentionLayer(
                attn=attn,
                mlp=mlp,
                ca_norm=RMSNorm(dim=embed_dim),
                mlp_norm=RMSNorm(dim=embed_dim),
                ca_scale=TanhGate(),
                mlp_scale=TanhGate(),
            )
            layers.append(FusionLayer(layer=decoder_layer, fusion_layer=xattn_layer))
        else:
            layers.append(decoder_layer)

    line_embeddings = nn.Linear(config['vocab_size'], config['embed_dim'], bias=False)

    decoder = TransformerDecoder(tok_embeddings=line_embeddings,
                                 layers=layers,
                                 max_seq_len=config['max_seq_len'],
                                 num_heads=config['num_heads'],
                                 head_dim=head_dim,
                                 norm=RMSNorm(config['embed_dim'], eps=1e-05),
                                 output=nn.Identity(),
                                 output_hidden_states=list(range(0, num_layers-1, 2)))

    if pretrained:
        weight_path = hf_hub_download(repo_id=pretrained, filename='model.safetensors')
        from safetensors import safe_open
        with safe_open(weight_path, framework='pt') as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


class OrliAdapter(nn.Module):
    """
    Builds an adapter head consisting of `num_layers` self attention layers
    followed by a linear projection of encoder_embed_dim to decoder_embed_dim.
    """
    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 encoder_embed_dims: list[int],
                 decoder_embed_dim: int):
        super().__init__()
        mlp_ratio = 4
        num_kv_heads = num_heads
        self.adapter = nn.ModuleList()

        for encoder_embed_dim in encoder_embed_dims:
            hidden_dim = int(mlp_ratio * encoder_embed_dim)
            head_dim = encoder_embed_dim // num_heads

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

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        os = []
        for idx, hidden_state in enumerate(encoder_hidden_states):
            hidden_state = hidden_state.flatten(-2).transpose(-1, -2)
            os.append(self.adapter[idx](hidden_state))
        return torch.cat(os, dim=1)


class CurveRegressionHead(nn.Module):
    """
    Iterative refinement regression head for baseline curves.
    """

    def __init__(self,
                 anchors: Union[str, 'PathLike'],
                 embed_dim: int = 576,
                 num_cls: int = 4,
                 num_layers: int = 3,
                 num_iterations: int = 4):
        super().__init__()
        reg_proj = nn.Sequential()
        self.register_buffer('curve_anchors', torch.load(anchors, map_location='cpu'))
        self.num_anchors = self.curve_anchors.shape[0]

        if num_layers > 1:
            for n in range(num_layers-1):
                reg_proj.append(nn.Linear(scale_hidden_dim_for_mlp(embed_dim) if n else embed_dim, scale_hidden_dim_for_mlp(embed_dim)))
                reg_proj.append(nn.SiLU())
        reg_proj.append(nn.Linear(scale_hidden_dim_for_mlp(embed_dim) if num_layers > 1 else embed_dim, self.num_anchors * 8))
        cls_proj = nn.Linear(embed_dim, self.num_anchors * num_cls)
        self.reg_projs = _get_clones(reg_proj, num_iterations)
        self.cls_projs = _get_clones(cls_proj, num_iterations)

    def forward(self, xs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Args:
            xs: A list containing `num_iterations` tensors of shape
                ``[b, s, d_e]`` where ``d_e`` is the decoder embedding dim.

        Returns:
            A dictionary containing an output tensor with shape ``[num_iterations, b, s, num_anchors, 8]``
            under the key `curves` and the class logits in `tokens` with shape
            ``[num_iterations, b, s, num_anchors, num_cls]``.
        """
        batch_size, seq_len, _ = xs[0].shape
        # expand anchors to batch and sequence dimensions
        _curves: list[torch.Tensor] = [self.curve_anchors.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)]
        _logits: list[torch.Tensor] = []
        for cls_proj, reg_proj, layer in zip(self.cls_projs, self.reg_projs, xs):
            curves = _curves[-1].detach()
            offsets = reg_proj(layer).view(batch_size, seq_len, self.num_anchors, 8)
            _curves.append(curves + offsets)
            _logits.append(cls_proj(layer).view(batch_size, seq_len, self.num_anchors, -1))

        return {'curves': torch.stack(_curves[1:]),
                'tokens': torch.stack(_logits)}
