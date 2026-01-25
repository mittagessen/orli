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
    Samples points from a batch of cubic Bézier curves.

    Args:
        curves (torch.Tensor): A tensor of shape (..., 8) containing the
                               control points of the Bézier curves in the
                               format (x0, y0, x1, y1, x2, y2, x3, y3).
        num_samples (int): The number of points to sample along each curve.

    Returns:
        A tensor of shape (..., num_samples, 2) containing the sampled points.
    """
    batch_dims = curves.shape[:-1]
    # Reshape curves to (..., 4, 2)
    p = curves.view(*batch_dims, 4, 2)
    p0, p1, p2, p3 = p[..., 0, :], p[..., 1, :], p[..., 2, :], p[..., 3, :]

    # Generate t values
    t = torch.linspace(0.0, 1.0, num_samples, device=curves.device)
    # expand t for batch operations
    # 1. unsqueeze to (1, num_samples)
    # 2. unsqueeze to (1, num_samples, 1) to allow broadcasting with (..., 1, 2)
    t = t.view(1, -1, 1)

    # expand dims of control points for broadcasting
    p0 = p0.unsqueeze(-2)
    p1 = p1.unsqueeze(-2)
    p2 = p2.unsqueeze(-2)
    p3 = p3.unsqueeze(-2)

    # Calculate points
    # (1-t)
    one_minus_t = 1 - t
    # B(t) = P0*(1-t)^3 + P1*3*t*(1-t)^2 + P2*3*t^2*(1-t) + P3*t^3
    points = (
        p0 * (one_minus_t**3) +
        p1 * (3 * t * one_minus_t**2) +
        p2 * (3 * t**2 * one_minus_t) +
        p3 * (t**3)
    )
    return points
