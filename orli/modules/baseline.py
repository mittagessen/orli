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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Baseline geometry helpers.

The trainable baseline representation is a bounded local-frame parameter
vector:

    cx, cy, length / sqrt(2), sin(theta), cos(theta), d_0, ..., d_{K-1}

``sin(theta)``, ``cos(theta)``, and the normal offsets ``d_i`` are stored in
``[0, 1]`` by affine remapping from ``[-1, 1]`` for directions and from
``[-0.5, 0.5]`` for offsets. Decoding produces a fixed-size polyline:

    p_i = c + (s_i - 0.5) * L * u + d_i * n
"""
from __future__ import annotations

import math

import torch

from orli.modules.bezier import sample_bezier_curve

DEFAULT_NUM_BASELINE_POINTS = 16
LOCAL_BASELINE_PARAM_DIM = 5


def baseline_param_dim(num_points: int = DEFAULT_NUM_BASELINE_POINTS) -> int:
    return LOCAL_BASELINE_PARAM_DIM + int(num_points)


def baseline_polyline_dim(num_points: int = DEFAULT_NUM_BASELINE_POINTS) -> int:
    return 2 * int(num_points)


def fixed_arc_length_resample(points: torch.Tensor,
                              num_points: int = DEFAULT_NUM_BASELINE_POINTS) -> torch.Tensor:
    """
    Resample polylines to a fixed number of approximately arc-length-uniform
    points.

    Args:
        points: Tensor of shape ``[..., n, 2]``.
        num_points: Number of output points.

    Returns:
        Tensor of shape ``[..., num_points, 2]``.
    """
    orig_shape = points.shape[:-2]
    flat = points.reshape(-1, points.shape[-2], 2)
    resampled = []

    for line in flat:
        if line.shape[0] < 2:
            resampled.append(line[:1].expand(num_points, 2))
            continue

        diffs = line[1:] - line[:-1]
        seg_lengths = torch.linalg.vector_norm(diffs, dim=-1)
        cum_lengths = torch.cat([torch.zeros(1, dtype=line.dtype, device=line.device),
                                 torch.cumsum(seg_lengths, dim=0)])
        total_length = cum_lengths[-1]
        if total_length <= 1e-8:
            resampled.append(line[:1].expand(num_points, 2))
            continue

        target_lengths = torch.linspace(0,
                                        total_length.item(),
                                        num_points,
                                        dtype=line.dtype,
                                        device=line.device)
        indices = torch.searchsorted(cum_lengths, target_lengths).clamp(1, line.shape[0] - 1)
        seg_start = cum_lengths[indices - 1]
        seg_end = cum_lengths[indices]
        denom = (seg_end - seg_start).clamp_min(1e-8)
        t = (target_lengths - seg_start) / denom
        p0 = line[indices - 1]
        p1 = line[indices]
        resampled.append(p0 + t.unsqueeze(-1) * (p1 - p0))

    return torch.stack(resampled, dim=0).reshape(*orig_shape, num_points, 2)


def bezier_to_polyline(curves: torch.Tensor,
                       num_points: int = DEFAULT_NUM_BASELINE_POINTS,
                       num_samples: int = 64) -> torch.Tensor:
    sampled = sample_bezier_curve(curves, num_samples=num_samples)
    return fixed_arc_length_resample(sampled, num_points=num_points)


def polyline_to_local_params(points: torch.Tensor) -> torch.Tensor:
    """
    Encodes fixed-size polylines as bounded local-frame baseline vectors.

    Args:
        points: Tensor of shape ``[..., K, 2]`` in normalized page coordinates.

    Returns:
        Tensor of shape ``[..., 5 + K]``.
    """
    num_points = points.shape[-2]
    start = points[..., 0, :]
    end = points[..., -1, :]
    chord = end - start
    length = torch.linalg.vector_norm(chord, dim=-1, keepdim=True)
    safe_length = length.clamp_min(1e-8)
    fallback = torch.zeros_like(chord)
    fallback[..., 0] = 1.0
    u = torch.where(length > 1e-8, chord / safe_length, fallback)
    n = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    center = (start + end) * 0.5

    s = torch.linspace(0,
                       1,
                       num_points,
                       dtype=points.dtype,
                       device=points.device)
    base = center.unsqueeze(-2) + (s - 0.5).view(*([1] * len(points.shape[:-2])), num_points, 1) * length.unsqueeze(-2) * u.unsqueeze(-2)
    offsets = ((points - base) * n.unsqueeze(-2)).sum(dim=-1)

    length01 = (length.squeeze(-1) / math.sqrt(2.0)).clamp(0.0, 1.0)
    sin01 = ((u[..., 1] + 1.0) * 0.5).clamp(0.0, 1.0)
    cos01 = ((u[..., 0] + 1.0) * 0.5).clamp(0.0, 1.0)
    offsets01 = (offsets + 0.5).clamp(0.0, 1.0)
    return torch.cat([center.clamp(0.0, 1.0),
                      length01.unsqueeze(-1),
                      sin01.unsqueeze(-1),
                      cos01.unsqueeze(-1),
                      offsets01],
                     dim=-1)


def local_params_to_polyline(params: torch.Tensor,
                             num_points: int | None = None) -> torch.Tensor:
    """
    Decodes local-frame baseline vectors to fixed-size polylines.

    Args:
        params: Tensor of shape ``[..., 5 + K]``.
        num_points: Optional number of decoded points. If omitted it is inferred
                    from the last dimension.

    Returns:
        Tensor of shape ``[..., K, 2]`` in normalized page coordinates.
    """
    if num_points is None:
        num_points = params.shape[-1] - LOCAL_BASELINE_PARAM_DIM
    if params.shape[-1] != baseline_param_dim(num_points):
        raise ValueError(f'Expected local baseline dim {baseline_param_dim(num_points)}, got {params.shape[-1]}.')

    center = params[..., :2]
    length = params[..., 2:3] * math.sqrt(2.0)
    sin = params[..., 3:4] * 2.0 - 1.0
    cos = params[..., 4:5] * 2.0 - 1.0
    norm = torch.sqrt((sin * sin + cos * cos).clamp_min(1e-8))
    u = torch.cat([cos / norm, sin / norm], dim=-1)
    n = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    offsets = params[..., LOCAL_BASELINE_PARAM_DIM:] - 0.5

    s = torch.linspace(0,
                       1,
                       num_points,
                       dtype=params.dtype,
                       device=params.device)
    shape_prefix = [1] * len(params.shape[:-1])
    base = center.unsqueeze(-2) + (s - 0.5).view(*shape_prefix, num_points, 1) * length.unsqueeze(-2) * u.unsqueeze(-2)
    return base + offsets.unsqueeze(-1) * n.unsqueeze(-2)


def curve_to_local_params(curves: torch.Tensor,
                          num_points: int = DEFAULT_NUM_BASELINE_POINTS,
                          num_samples: int = 64) -> torch.Tensor:
    points = bezier_to_polyline(curves, num_points=num_points, num_samples=num_samples)
    return polyline_to_local_params(points)


def prepare_baseline_anchors(anchors: torch.Tensor,
                             num_points: int = DEFAULT_NUM_BASELINE_POINTS) -> torch.Tensor:
    """
    Converts anchor tables in legacy cubic, fixed-polyline, or local-param form
    to local-frame baseline parameters.
    """
    anchors = anchors.float()
    target_dim = baseline_param_dim(num_points)
    if anchors.shape[-1] == target_dim:
        return anchors.clamp(1e-5, 1.0 - 1e-5)
    if anchors.shape[-1] == 8:
        return curve_to_local_params(anchors, num_points=num_points).clamp(1e-5, 1.0 - 1e-5)
    if anchors.shape[-1] == baseline_polyline_dim(num_points):
        points = anchors.reshape(*anchors.shape[:-1], num_points, 2)
        return polyline_to_local_params(points).clamp(1e-5, 1.0 - 1e-5)
    raise ValueError(f'Unsupported anchor width {anchors.shape[-1]}; expected 8, {baseline_polyline_dim(num_points)}, or {target_dim}.')
