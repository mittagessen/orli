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
from typing import Generator, Optional, Union, List, Dict, TYPE_CHECKING, Sequence

import json
import torch
import torch.nn.functional as F
from torch import nn

from orli.modules import (MultiHeadAttention, RMSNorm, TanhGate,
                          TransformerCrossAttentionLayer,
                          TransformerDecoder, TransformerSelfAttentionLayer,
                          FusionLayer, scale_hidden_dim_for_mlp,
                          Llama3ScaledRoPE, ScaleEncoder, llama3_mlp)

if TYPE_CHECKING:
    from os import PathLike

logger = logging.getLogger(__name__)

__all__ = ['baseline_decoder', 'OrliModel']


def baseline_decoder(vocab_size: int = 12,
                     num_layers: int = 4,
                     num_heads: int = 9,
                     num_kv_heads: int = 3,
                     embed_dim: int = 576,
                     max_seq_len: int = 768,
                     intermediate_dim: int = 1536,
                     attn_dropout: int = 0.0,
                     norm_eps: int = 1e-5,
                     rope_base: int = 10000,
                     encoder_max_seq_len: int = 12544,  # start of fusion parameters
                     pretrained: Optional[str] = None,
                     **kwargs) -> TransformerDecoder:
    """
    Builds a decoder regression baselines.

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
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
            k_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
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
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(config['embed_dim'], config['num_heads'] * head_dim, bias=False),
            k_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(config['embed_dim'], num_kv_heads * head_dim, bias=False),
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

    line_embeddings = nn.Linear(config['vocab_size'], config['embed_dim'], bias=False)
    output_proj = CurveHead(config['embed_dim'])

    decoder = TransformerDecoder(tok_embeddings=line_embeddings,
                                 layers=layers,
                                 max_seq_len=config['max_seq_len'],
                                 num_heads=config['num_heads'],
                                 head_dim=head_dim,
                                 norm=RMSNorm(config['embed_dim'], eps=1e-05),
                                 output=output_proj)

    if pretrained:
        weight_path = hf_hub_download(repo_id=pretrained, filename='model.safetensors')
        from safetensors import safe_open
        with safe_open(weight_path, framework='pt') as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
        decoder.load_state_dict(state_dict, strict=False)

    return decoder


class PruningModule(nn.Module):
    """
    Learned selective token pruning module.

    Uses a simple MLP to produce a score map selecting `topk` tokens from an
    input tensor with shape (B, S, C).

    Args:
        channels: # of channels `C` of input
        hidden_dim: dimensionality of intermediate MLP layer
        topk: number of tokens to select.
    """
    def __init__(self,
                 channels: int,
                 intermediate_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(RMSNorm(channels, eps=1e-05),
                                 nn.Linear(channels, intermediate_dim, bias=False),
                                 nn.SiLU(),
                                 nn.Linear(intermediate_dim, 1, bias=False))

    def forward(self, x: torch.Tensor, topk: int):
        sel = self.mlp(x)  # (b,l,1)
        # select tokens
        _, topk_ind = torch.topk(sel, topk, dim=-2)
        return x.gather(dim=1, index=topk_ind.repeat(1, 1, x.shape[-1]))


class EncoderFusion(nn.Module):
    """
    Fuses encoder feature pyramids with selective token pruning.

    Args:
        in_channels: Channel dimension size of each feature map.
        topk_tokens: Number of tokens to retain for each map.
        embed_dim: Output token embedding dimension.
        intermediate_dim: Intermediate dimension of token pruning layer.
    """
    def __init__(self,
                 in_channels: List[int],
                 topk_tokens: List[int],
                 embed_dim: int = 128,
                 intermediate_dim: int = 768):
        super().__init__()
        self.topk_tokens = topk_tokens

        self.scale_encoder = ScaleEncoder(len(in_channels))

        self.output_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.output_proj.append(nn.Linear(in_channel, embed_dim, bias=False))

        self.pruning_layers = nn.ModuleList()
        for idx, channels in enumerate(in_channels):
            self.pruning_layers.append(PruningModule(channels, intermediate_dim))

    def forward(self, features: Sequence[torch.Tensor]):
        """
        Args:
            features: List of feature pyramid tensors with shape ``[b x c x h_n x w_n]``

        Returns:
            Tensor: output tensor with shape ``[b x t x e]``
        Notation used for tensor shapes:
            - b: batch size
            - c: channels
            - h_n: height of n-th feature map
            - w_n: width of n-th feature map
            - t: sum of lengths of topk_tokens
            - e: embed dim
        """
        os = []
        for feat, prune, proj, tkt in zip(features,
                                          self.pruning_layers,
                                          self.output_proj,
                                          self.topk_tokens):
            o = prune(feat.flatten(-2).transpose(1, 2), tkt)  # NCHW -> N(TKT)C
            os.append(proj(o))
        os = self.scale_encoder(os)
        return torch.cat(os, dim=1)


class CurveHead(nn.Module):
    """
    Classification and regression head for baseline curves.
    """
    def __init__(self,
                 embed_dim: int = 576,
                 num_cls: int = 4):
        super().__init__()
        self.cls_proj = nn.Linear(embed_dim, num_cls, bias=False)
        self.reg_proj = nn.Linear(embed_dim, 8, bias=False)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return {'tokens': self.cls_proj(x),
                'curves': self.reg_proj(x).sigmoid()}


class OrliModel(nn.Module):
    """
    The transformer segmentation fusion model.

    Args:
        encoder: A timm image encoder model
        adapter:
        decoder: Curve decoder model
    """
    eos_id: int = 2

    def __init__(self,
                 encoder: nn.Module,
                 adapter: nn.Module,
                 decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.decoder = decoder
        self.ready_for_generation = False

    @classmethod
    def from_safetensors(cls, filename: Union[str, 'PathLike']) -> 'OrliModel':
        """
        Loads model weights from a safetensors-based kraken serialization.
        """
        import timm
        from safetensors import safe_open

        with safe_open(filename, framework='pt') as f:
            metadata = f.metadata()
            config = json.loads(metadata['config'])
            encoder_config = {k[8:]: v for k, v in config.items() if k.startswith('encoder_')}
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        out_indices = list(range(4 - len(encoder_config['topk_tokens']), 4, 1))
        max_seq_len = sum(encoder_config['topk_tokens'])

        encoder_model = timm.create_model(encoder_config['name'],
                                          pretrained=False,
                                          features_only=True,
                                          out_indices=out_indices)

        adapter = EncoderFusion(in_channels=encoder_model.feature_info.channels(),
                                topk_tokens=encoder_config['topk_tokens'],
                                embed_dim=encoder_config['embed_dim'])

        decoder_model = baseline_decoder(embed_dim=encoder_config['embed_dim'],
                                         encoder_max_seq_len=max_seq_len)

        model = cls(encoder=encoder_model,
                    adapter=adapter,
                    decoder=decoder_model)

        model.load_state_dict(state_dict, strict=False)

        return model

    def setup_caches(self,
                     batch_size: int,
                     dtype: torch.dtype,
                     *,
                     encoder_max_seq_len: int = None,
                     decoder_max_seq_len: int = None):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`party.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`party.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`party.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum encoder cache sequence length.
            decoder_max_seq_len (int): maximum decoder cache sequence length.
        """
        self.decoder.setup_caches(batch_size,
                                  dtype,
                                  encoder_max_seq_len=encoder_max_seq_len,
                                  decoder_max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.decoder.caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.decoder.caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.decoder.reset_caches()

    def forward(self,
                tokens: torch.Tensor,
                *,
                encoder_input: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input: Optional input for the encoder.
            encoder_hidden_states: Optional encoder embeddings with curve
                                   embeddings already added.
            encoder_curves: Optional curves to be embedded and added to encoder
                            embeddings.
            input_pos: Optional tensor which contains the position ids of each
                       token. During training, this is used to indicate the
                       positions of each token relative to its sample when
                       packed, shape ``[b x s]``.  During inference, this
                       indicates the position of the current token.  If none,
                       assume the index of the token is its position id.
                       Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        if encoder_input is not None:
            encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)

        output = self.decoder(tokens=tokens,
                              mask=mask,
                              encoder_input=encoder_hidden_states,
                              encoder_mask=encoder_mask,
                              input_pos=input_pos)
        return output

    def forward_encoder_embeddings(self, encoder_input):
        """
        Computes the encoder embeddings *without* adding the curve positional
        embeddings.
        """
        encoder_hidden_states = self.encoder(encoder_input)
        return self.adapter(encoder_hidden_states)

    @torch.inference_mode()
    def prepare_for_generation(self,
                               bos_id: torch.Tensor = torch.Tensor([1]).long(),
                               batch_size: int = 1,
                               max_encoder_seq_len: int = 12544,
                               max_generated_tokens: int = 768,
                               device: torch.device = torch.device('cpu')):

        if self.ready_for_generation:
            raise ValueError('Model has already been prepared for generation!')

        self._batch_size = batch_size
        self._max_generated_tokens = max_generated_tokens
        self._max_encoder_seq_len = max_encoder_seq_len

        # generate a regular causal mask
        self._masks = torch.tril(torch.ones(max_generated_tokens,
                                            max_generated_tokens,
                                            dtype=torch.bool,
                                            device=device)).unsqueeze(0)
        self._input_pos = torch.arange(0, max_generated_tokens, device=device).unsqueeze(0)

        # set up caches
        self.setup_caches(batch_size=batch_size,
                          encoder_max_seq_len=max_encoder_seq_len,
                          decoder_max_seq_len=max_generated_tokens,
                          dtype=next(self.encoder.parameters()).dtype)

        # create batch size number of BOS tokens
        self._prompt = F.one_hot(bos_id, num_classes=12).to(device=device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1, 1)
        self.ready_for_generation = True

    @torch.inference_mode()
    def predict(self,
                encoder_input: torch.FloatTensor) -> Generator[torch.Tensor, None, None]:
        """
        Predicts Bézier curves and line classes.

        Args:
            encoder_input: Image input for the encoder with shape ``n x c x h x w``

        Returns:
            A float tensor of with shape ``n x s x 8`` where ``n`` is the batch
            size, ``s`` is the maximum number of curves detected, and the last
            dimension is an 8-tuple defining the control points of a cubic
            Bézier curve. No-output is indicated by zeroed output control
            points. If s < max_generated_tokens the last entry across all batch
            items will be a no-output.
        """
        logger.info('Computing encoder embeddings')

        encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)

        eos_token = torch.tensor(self.eos_id, device=encoder_hidden_states.device, dtype=torch.long)

        # Mask is shape (batch_size, max_seq_len, image_embedding_len)
        encoder_mask = torch.ones((encoder_hidden_states.size(0),
                                   1,
                                   encoder_hidden_states.size(1)),
                                  dtype=torch.bool,
                                  device=encoder_input.device)
        # prefill step
        curr_masks = self._masks[:, :1]
        logits = self.forward(tokens=self._prompt,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_mask=encoder_mask,
                              mask=curr_masks,
                              input_pos=self._input_pos[:, :1].squeeze())
        tokens = torch.argmax(logits['tokens'], dim=-1)
        curves = logits['curves']
        generated_tokens = [tokens[:, -1]]
        generated_curves = [curves[:, -1]]

        curr_pos = 1

        # keeps track of EOS tokens emitted by each sequence in a batch
        eos_token_reached = torch.zeros(self._batch_size, dtype=torch.bool, device=encoder_input.device)
        eos_token_reached |= tokens[:, -1] == eos_token

        # mask used for setting all values from EOS token to pad_id in output sequences.
        eos_token_mask = torch.ones(self._batch_size, 0, dtype=torch.int32, device=curves.device)

        for _ in range(self._max_generated_tokens - 1):
            # update eos_token_mask if an EOS token was emitted in a previous step
            eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(self._batch_size, 1)], dim=-1)

            curr_input_pos = self._input_pos[:, curr_pos]
            curr_masks = self._masks[:, curr_pos, None, :]

            # no need for encoder embeddings anymore as they're in the cache now
            logits = self.forward(tokens=torch.cat([F.one_hot(tokens, num_classes=4).to(device=encoder_hidden_states.device),
                                                    curves], dim=-1),
                                  mask=curr_masks,
                                  input_pos=curr_input_pos)
            print(f'tokens: {logits["tokens"]} curves: {logits["curves"]}')
            tokens = torch.argmax(logits['tokens'], dim=-1)
            curves = logits['curves']
            generated_tokens.append(tokens[:, -1])
            generated_curves.append(curves[:, -1])

            curr_pos += 1

            eos_token_reached |= tokens[:, -1] == eos_token
            logger.info(f'Generated tokens: {tokens} Generated curves: {curves}')

            if eos_token_reached.all():
                break

        eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(self._batch_size, 1)], dim=-1)
        eos_curve_mask = eos_token_mask.expand(8, -1, -1).permute(1, 2, 0)
        return torch.stack(generated_curves, dim=1) * eos_curve_mask
