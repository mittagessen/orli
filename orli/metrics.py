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
orli.metrics
~~~~~~~~~~~~

Baseline detection and reading order evaluation metrics.

Detection metrics are adapted from the Transkribus Baseline Evaluation
Scheme, using optimal (Hungarian) matching instead of greedy matching.
"""
import torch
import logging

from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

from orli.modules.baseline import (LOCAL_BASELINE_PARAM_DIM,
                                   local_params_to_polyline)

logger = logging.getLogger(__name__)

__all__ = ['evaluate_page', 'aggregate_metrics']


def interpolate_polyline(points: torch.Tensor, spacing: float = 5.0) -> torch.Tensor:
    """
    Resample a polyline to approximately uniform point spacing.

    Args:
        points: Tensor of shape (N, 2) representing polyline vertices.
        spacing: Target distance between consecutive points in pixels.

    Returns:
        Tensor of shape (M, 2) with uniformly spaced points.
    """
    if points.shape[0] < 2:
        return points

    diffs = points[1:] - points[:-1]
    seg_lengths = torch.norm(diffs, dim=-1)
    cum_lengths = torch.cat([torch.zeros(1, device=points.device, dtype=points.dtype),
                             torch.cumsum(seg_lengths, dim=0)])
    total_length = cum_lengths[-1]

    if total_length < 1e-6:
        return points[:1]

    num_points = max(2, int(torch.round(total_length / spacing).item()))
    target_lengths = torch.linspace(0, total_length.item(), num_points,
                                    device=points.device, dtype=points.dtype)

    indices = torch.searchsorted(cum_lengths, target_lengths).clamp(1, len(cum_lengths) - 1)

    seg_start = cum_lengths[indices - 1]
    seg_end = cum_lengths[indices]
    seg_len = seg_end - seg_start
    t = torch.where(seg_len > 1e-8,
                    (target_lengths - seg_start) / seg_len,
                    torch.zeros_like(target_lengths))

    p0 = points[indices - 1]
    p1 = points[indices]
    return p0 + t.unsqueeze(-1) * (p1 - p0)


def _point_scores(min_dists: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Per-point scoring.

    Args:
        min_dists: Minimum distances from each point to the other polyline.
        tol: Tolerance in pixels.

    Returns:
        Scores in [0, 1] for each point.
    """
    return torch.where(
        min_dists <= tol,
        torch.ones_like(min_dists),
        torch.where(
            min_dists < 3 * tol,
            (3 * tol - min_dists) / (2 * tol),
            torch.zeros_like(min_dists),
        ),
    )


def baseline_score(pred_points: torch.Tensor,
                   gt_points: torch.Tensor,
                   tol: float) -> float:
    """
    Directed score from one polyline to another.

    For each point in pred, finds the minimum distance to any point in gt,
    applies the point score, and returns the mean.

    Args:
        pred_points: (M, 2) uniformly spaced points on prediction polyline.
        gt_points: (N, 2) uniformly spaced points on GT polyline.
        tol: Tolerance in pixels.

    Returns:
        Mean point score (directed, pred -> gt).
    """
    dists = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)
    min_dists = dists.min(dim=1).values
    return _point_scores(min_dists, tol).mean().item()


def match_baselines(pred_polylines: list[torch.Tensor],
                    gt_polylines: list[torch.Tensor],
                    tol: float) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
    """
    Build a symmetric score matrix and solve the optimal assignment.

    Args:
        pred_polylines: List of P predicted polylines, each (M_i, 2).
        gt_polylines: List of G ground truth polylines, each (N_j, 2).
        tol: Tolerance in pixels.

    Returns:
        score_matrix: (P, G) symmetric baseline scores.
        matches: List of (pred_idx, gt_idx) pairs from assignment.
        match_scores: Scores for each matched pair.
    """
    n_pred = len(pred_polylines)
    n_gt = len(gt_polylines)

    score_matrix = torch.zeros(n_pred, n_gt)
    for i, pred in enumerate(pred_polylines):
        for j, gt in enumerate(gt_polylines):
            s_pg = baseline_score(pred, gt, tol)
            s_gp = baseline_score(gt, pred, tol)
            score_matrix[i, j] = (s_pg + s_gp) / 2.0

    cost_matrix = 1.0 - score_matrix.numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    match_scores = score_matrix[row_ind, col_ind]

    return score_matrix, matches, match_scores


def compute_detection_metrics(pred_polylines: list[torch.Tensor],
                              gt_polylines: list[torch.Tensor],
                              tol: float) -> dict[str, float]:
    """
    Compute precision, recall, and F1 for one page.

    Args:
        pred_polylines: List of P predicted polylines.
        gt_polylines: List of G GT polylines.
        tol: Tolerance in pixels.

    Returns:
        Dict with 'precision', 'recall', 'f1', 'num_pred', 'num_gt'.
    """
    n_pred = len(pred_polylines)
    n_gt = len(gt_polylines)

    if n_pred == 0 and n_gt == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'num_pred': 0, 'num_gt': 0}
    if n_pred == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'num_pred': 0, 'num_gt': n_gt}
    if n_gt == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'num_pred': n_pred, 'num_gt': 0}

    _, matches, match_scores = match_baselines(pred_polylines, gt_polylines, tol)

    precision = match_scores.sum().item() / n_pred
    recall = match_scores.sum().item() / n_gt

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'num_pred': n_pred, 'num_gt': n_gt}


def compute_ordering_metrics(matches: list[tuple[int, int]],
                             match_scores: torch.Tensor,
                             match_threshold: float = 0.5) -> dict[str, float]:
    """
    Compute reading-order metrics on matched baselines.

    After filtering matches by a minimum score threshold, extracts rank
    vectors and computes:
      - Spearman's footrule distance (normalized to [0, 1])
      - Kendall's tau correlation (in [-1, 1])

    Args:
        matches: List of (pred_idx, gt_idx) pairs.
        match_scores: Score for each matched pair.
        match_threshold: Minimum match score for inclusion.

    Returns:
        Dict with 'spearman_footrule', 'kendall_tau',
        'num_matched_for_ordering'.
    """
    # Filter to well-matched pairs
    good = [(p, g) for (p, g), s in zip(matches, match_scores)
            if s.item() >= match_threshold]

    if len(good) < 2:
        return {'spearman_footrule': float('nan'),
                'kendall_tau': float('nan'),
                'num_matched_for_ordering': len(good)}

    # Sort by GT index and re-rank both to 0..n-1
    good.sort(key=lambda x: x[1])
    gt_ranks = list(range(len(good)))
    # Pred indices in the order sorted by GT
    pred_indices = [p for p, _ in good]
    # Convert to ranks (rank of each pred_index in sorted order)
    pred_rank_order = sorted(range(len(pred_indices)), key=lambda i: pred_indices[i])
    pred_ranks = [0] * len(pred_indices)
    for rank, idx in enumerate(pred_rank_order):
        pred_ranks[idx] = rank

    n = len(gt_ranks)

    # Spearman's footrule: sum |pred_rank[i] - gt_rank[i]| / max_displacement
    footrule = sum(abs(pred_ranks[i] - gt_ranks[i]) for i in range(n))
    max_footrule = n * n // 2  # floor(n^2/2)
    norm_footrule = footrule / max_footrule if max_footrule > 0 else 0.0

    # Kendall's tau
    tau, _ = kendalltau(pred_ranks, gt_ranks)

    return {'spearman_footrule': norm_footrule,
            'kendall_tau': tau,
            'num_matched_for_ordering': n}


def _baseline_vectors_to_polylines(curves: torch.Tensor,
                                   image_size: tuple[int, int],
                                   spacing: float = 5.0,
                                   num_baseline_points: int | None = None) -> list[torch.Tensor]:
    """
    Convert baseline vectors to uniformly-spaced pixel polylines.

    Args:
        curves: ``(S, D)`` normalized baseline vectors. Supported formats are
                local-frame baseline parameters (D=5+K points) and flat fixed
                polylines (D=2*K points).
        image_size: ``(width, height)`` used for denormalization.
        spacing: Target spacing for interpolation.

    Returns:
        List of (M_i, 2) tensors, one per curve.
    """
    if curves.numel() == 0:
        return []

    w, h = image_size
    point_scale = torch.tensor([w, h], dtype=curves.dtype, device=curves.device)

    if num_baseline_points is not None and curves.shape[-1] == 2 * num_baseline_points:
        sampled = curves.reshape(curves.shape[0], curves.shape[-1] // 2, 2) * point_scale
    elif curves.shape[-1] > LOCAL_BASELINE_PARAM_DIM:
        sampled = local_params_to_polyline(curves, num_points=num_baseline_points).clamp(0.0, 1.0) * point_scale
    elif curves.shape[-1] % 2 == 0:
        sampled = curves.reshape(curves.shape[0], curves.shape[-1] // 2, 2) * point_scale
    else:
        raise ValueError(f'Unsupported baseline vector width {curves.shape[-1]}.')

    polylines = []
    for i in range(sampled.shape[0]):
        polylines.append(interpolate_polyline(sampled[i], spacing))
    return polylines


def evaluate_page(pred_curves: torch.Tensor,
                  gt_curves: torch.Tensor,
                  image_size: tuple[int, int],
                  tol: float,
                  match_threshold: float = 0.5,
                  spacing: float = 5.0) -> dict[str, float]:
    """
    Full per-page evaluation.

    Args:
        pred_curves: Predicted normalized baseline vectors (zero rows already
                     filtered).
        gt_curves: Ground-truth normalized baseline vectors or flat fixed
                   polylines.
        image_size: (width, height) of the original image.
        tol: Tolerance in pixels.
        match_threshold: Minimum score for ordering evaluation.
        spacing: Polyline interpolation spacing.

    Returns:
        Combined dict of detection and ordering metrics.
    """
    num_baseline_points = None
    if pred_curves.shape[-1] > LOCAL_BASELINE_PARAM_DIM:
        num_baseline_points = pred_curves.shape[-1] - LOCAL_BASELINE_PARAM_DIM

    pred_polylines = _baseline_vectors_to_polylines(pred_curves,
                                                    image_size,
                                                    spacing,
                                                    num_baseline_points)
    gt_polylines = _baseline_vectors_to_polylines(gt_curves,
                                                  image_size,
                                                  spacing,
                                                  num_baseline_points)

    det = compute_detection_metrics(pred_polylines, gt_polylines, tol)

    # Ordering uses the matching from detection
    if len(pred_polylines) > 0 and len(gt_polylines) > 0:
        _, matches, match_scores = match_baselines(pred_polylines, gt_polylines, tol)
        order = compute_ordering_metrics(matches, match_scores, match_threshold)
    else:
        order = {'spearman_footrule': float('nan'),
                 'kendall_tau': float('nan'),
                 'num_matched_for_ordering': 0}

    return {**det, **order}


def aggregate_metrics(page_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Macro-average per-page metrics across the dataset.

    Ordering metrics skip NaN pages (fewer than 2 matched lines).

    Args:
        page_metrics: List of per-page metric dicts.

    Returns:
        Aggregated metric dict.
    """
    if not page_metrics:
        return {}

    n = len(page_metrics)

    precision = sum(m['precision'] for m in page_metrics) / n
    recall = sum(m['recall'] for m in page_metrics) / n
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    avg_num_pred = sum(m['num_pred'] for m in page_metrics) / n
    avg_num_gt = sum(m['num_gt'] for m in page_metrics) / n

    # Ordering: skip NaN pages
    footrule_vals = [m['spearman_footrule'] for m in page_metrics
                     if m['spearman_footrule'] == m['spearman_footrule']]  # NaN != NaN
    tau_vals = [m['kendall_tau'] for m in page_metrics
                if m['kendall_tau'] == m['kendall_tau']]

    spearman_footrule = sum(footrule_vals) / len(footrule_vals) if footrule_vals else float('nan')
    kendall_tau_val = sum(tau_vals) / len(tau_vals) if tau_vals else float('nan')

    return {'precision': precision,
            'recall': recall,
            'f1': f1,
            'spearman_footrule': spearman_footrule,
            'kendall_tau': kendall_tau_val,
            'avg_num_pred': avg_num_pred,
            'avg_num_gt': avg_num_gt,
            'num_pages': n}
