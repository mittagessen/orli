# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from typing import List, Optional

from torch import nn
from enum import Enum
from functools import partial
from typing import List, Optional

from .feed_forward import FeedForward
from .norms import Fp32LayerNorm

def llama3_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Llama model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)


class Llama3VisionProjectionHead(nn.Module):
    """Projection transformer to adapt the output of a
    pretrained frozen encoder (CLIP) to a pretrained decoder model.
    For example, nn.Sequential(CLIP(), Llama3VisionProjectionHead()).

    Args:
        layers (nn.Module): Transformer Decoder layers
        output (nn.Module): Output linear layer. Input dim is
            (num_hidden + 1) * encoder_dim and output is decoder_dim.
        num_hidden_inputs (int): Number of expected hidden state inputs
    """

    def __init__(
        self,
        layers: nn.Module,
        output: nn.Module,
        num_hidden_inputs: int = 0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.output = output
        self.num_hidden = num_hidden_inputs

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x i x t x e x d]
            hidden_states (Optional[List[torch.Tensor]]): list of hidden states
                from the encoder. Each hidden state has the same shape as x.

        Returns:
            Tensor: output tensor of a sequence of embedings [b x s x d]
                where sequence length is num_imgs*num_tiles+num_embeds

        Notation used for tensor shapes:
            - b: batch size
            - i: number of images
            - t: number of tiles (where a single image is broken into multiple tiles)
            - e: number of embeds per tile (e.g. CLS embed + patch embeds, etc.)
            - s: sequence length computed by i*t*e
            - d: embed dim
        """
        bsz, imgs, tiles, embeds, dim = x.shape

        # apply transformer layers
        x = x.view(bsz * imgs, tiles * embeds, dim)
        for layer in self.layers:
            x = layer(x)
        x = x.view(bsz, imgs, tiles, embeds, dim)

        # interleave hidden states and cat with x
        if self.num_hidden > 0:
            hidden_states = torch.stack(hidden_states, dim=-1)
            hidden_states = hidden_states.view(bsz, imgs, tiles, embeds, -1)
            x = torch.cat([x, hidden_states], dim=-1)

        # shape [b x s x d]
        x = self.output(x).reshape(bsz, imgs * tiles * embeds, -1)

        return x


def llama3_2_vision_projection_head(
    *,
    num_layers: int,
    num_heads: int,
    decoder_embed_dim: int,
    clip_embed_dim: int,
    num_hidden_inputs: int,
) -> Llama3VisionProjectionHead:
    """
    Build the Llama 3.2 Vision Projection Head that maps the output of the CLIP encoder
    to the decoder cross attention input.

    Args:
        num_layers (int): number of layers in the projection head.
        num_heads (int): number of heads in the projection head.
        decoder_embed_dim (int): embedding dimension for the decoder.
        clip_embed_dim (int): embedding dimension for the CLIP encoder.
        num_hidden_inputs (int): number of hidden inputs to the projection head.

    Returns:
        Llama3VisionProjectionHead: Instantiation of Llama 3.2 vision projection head.
    """
    mlp_ratio = 4
    hidden_dim = int(mlp_ratio * clip_embed_dim)
    head_dim = clip_embed_dim // num_heads
    num_kv_heads = num_heads

    layers = []
    for _ in range(num_layers):
        self_attn = MultiHeadAttention(
            embed_dim=clip_embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(clip_embed_dim, num_heads * head_dim, bias=False),
            k_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            v_proj=nn.Linear(clip_embed_dim, num_kv_heads * head_dim, bias=False),
            output_proj=nn.Linear(clip_embed_dim, clip_embed_dim, bias=False),
            pos_embeddings=None,
            attn_dropout=0.0,
            is_causal=False,
        )

        mlp = clip_mlp(
            in_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            out_dim=clip_embed_dim,
            activation=nn.GELU(),
        )

        layer = TransformerSelfAttentionLayer(
            attn=self_attn,
            mlp=mlp,
            sa_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            mlp_norm=Fp32LayerNorm(clip_embed_dim, eps=1e-5),
            sa_scale=TanhGate(),
            mlp_scale=TanhGate(),
        )
        layers.append(layer)

    # we concatenate clip embeddings and hidden layers output
    # and project it to embed_dim_out, which will be used for the
    # cross encoding
    proj_in = clip_embed_dim * (num_hidden_inputs + 1)
    return Llama3VisionProjectionHead(
        layers=layers,
        output=nn.Linear(proj_in, decoder_embed_dim),
        num_hidden_inputs=num_hidden_inputs,
    )


def scale_hidden_dim_for_mlp(dim: int, multiple_of: int = 256) -> int:
    """Scale hidden dimension for MLP to keep number of parameters and computation constant.

    Args:
        dim (int): Input dimension.
        multiple_of (int): Round scaled dimension to nearest multiple of `multiple_of` for clean computation.

    Returns:
        Scaled hidden dimension.
    """
    # Scale hidden dimension by (2/3)4d for SwiGLU to keep number of
    # parameters and computation constant
    hidden_dim = 4 * int(2 * dim / 3)
    # Round hidden dimension to nearest multiple of `multiple_of`
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class Llama3ScaledRoPE(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864 with additional
    scaling from https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1.

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Default scaling factors are from the following Meta-Llama code:
    https://github.com/meta-llama/llama-models/blob/dc42f22a3b05502e7296402b019a51f57fa045c9/models/llama3_1/api/model.py#L41

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
        scale_factor (int): scaling factor for theta. Default: 8
        low_freq_factor (int): low frequency factor for scaling theta. Default: 1
        high_freq_factor (int): high frequency factor for scaling theta. Default: 4
        old_context_len (int): old context length for scaling theta. Default: 8192
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
        scale_factor: int = 8,
        low_freq_factor: int = 1,
        high_freq_factor: int = 4,
        old_context_len: int = 8192,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        self.scale_factor = scale_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = old_context_len
        self.is_cache_built = False
        self.rope_init()

    def rope_init(self):
        """
        Warning: this is called in recipes before torch.compile,
        so that the cache is built in advance.
        """
        freqs = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )

        # If we're on meta device return early.
        # We can't apply scaling until freqs is filled with real data
        if freqs.is_meta:
            return

        theta = self.apply_scaling(
            freqs,
            self.scale_factor,
            self.low_freq_factor,
            self.high_freq_factor,
            self.old_context_len,
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)
        self.is_cache_built = True

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def apply_scaling(
        self,
        freqs: torch.Tensor,
        scale_factor: int,
        low_freq_factor: int,
        high_freq_factor: int,
        old_context_len: int,
    ):

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / scale_factor)
            else:
                assert low_freq_wavelen != high_freq_wavelen
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        Raises:
            RuntimeError: if RoPE cache is not initialized prior to forward call
        """

        if not self.is_cache_built:
            raise RuntimeError(
                "RoPE cache is not built. Please call rope_init() first."
            )

        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)
