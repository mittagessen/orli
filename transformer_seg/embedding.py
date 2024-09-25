# coding=utf-8
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
"""Bézier curve embedding."""

import logging
from typing import Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ['MBartScaledCurveEmbedding']


class MBartScaledCurveEmbedding(nn.EmbeddingBag):
    """
    Embedding bag for quadratic Bézier curve prompts for decoder.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, mode='sum')
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        """
        Maps an input tensor of (N, S, 8) to (N, S, embedding_dim)
        """
        b, s, p = input_ids.shape
        o = super().forward(input_ids.view(-1, p)) * self.embed_scale
        return o.view(b, s, -1)
