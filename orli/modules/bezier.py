#
# Copyright 2026 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Pytorch implementations of bezier curve sampling.
"""
import torch


def sample_bezier_curve(curves: torch.Tensor, num_samples: int = 20) -> torch.Tensor:
    """
    Evaluates cubic Bezier curves at a specific number of steps.

    Args:
        control_points: A tensor of shape [..., 8] containing the curve
        control points.
        num_samples: The number of points to sample along the curve.

    Returns:
        A tensor of shape [..., num_samples, 2] containing the sampled points.
    """
    p = curves.view(-1, 4, 2)
    t = torch.linspace(0, 1, num_samples, dtype=curves.dtype, device=curves.device)

    one_minus_t = 1 - t

    b0 = one_minus_t ** 3
    b1 = 3 * (one_minus_t ** 2) * t
    b2 = 3 * one_minus_t * (t ** 2)
    b3 = t ** 3

    coeffs = torch.stack([b0, b1, b2, b3], dim=-1)

    return torch.einsum('... i d, s i -> ... s d', p, coeffs)
