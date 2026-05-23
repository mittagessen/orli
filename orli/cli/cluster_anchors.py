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
orli.cli.cluster_anchors
~~~~~~~~~~~~~~~~~~~~~~~~

Re-derive the curve-anchor table from training data in chord-local space.

The default ``orli/assets/anchors.json`` was obtained by k-means in
image-space cubic-Bezier coordinates. Empirically this collapses to nearly
straight templates because the (P0, P3) chord-position component dominates
distance, so the curved tail of the corpus (Yiddish, Arabic, Hebrew,
Persian, Ottoman, warped manuscripts) contributes no curved prototypes.

This command re-clusters in the chord-local layout
``(P0, P3, t1, n1, t2, n2)``, optionally stratifying by shard so each
domain contributes its own cluster centres regardless of corpus mass. The
output file is written in image-space cubic-Bezier form so it can be
loaded by the existing ``OrliSegmentationTrainingConfig`` machinery
without any further plumbing.
"""
import json
import logging
from pathlib import Path

import click

from .util import _expand_gt, _validate_manifests, message

logger = logging.getLogger('orli')


def _read_curves_from_arrow(path: str):
    import pyarrow as pa
    curves = []
    with pa.memory_map(path, 'rb') as src:
        try:
            table = pa.ipc.open_file(src).read_all()
        except Exception:
            logger.warning(f'{path} is not an arrow file, skipping')
            return curves
        for i in range(len(table)):
            page = table.column('pages')[i].as_py()
            for line in page['lines']:
                curves.append(line['curve'])
    return curves


def _cluster_chord_local(curves_cl, k, seed: int = 0):
    """
    k-means in the 8-D chord-local space, with per-dim std normalization so
    the small-magnitude curvature coordinates ``(n1, n2)`` are not drowned
    out by chord placement during distance computation. Returns
    ``[k, 8]`` centres in unnormalized chord-local form.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    if len(curves_cl) < k:
        raise click.ClickException(
            f'Not enough samples ({len(curves_cl)}) for k={k}. Reduce k or'
            ' supply more data.')
    arr = np.asarray(curves_cl, dtype=np.float32)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = arr / std
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(normalized)
    dists = ((normalized[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)  # [N, k]
    medoid_idx = dists.argmin(axis=0)
    return arr[medoid_idx]   # real curves, not averages


@click.command('cluster-anchors')
@click.option('-t', '--training-files', 'training_data', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='Manifest file(s) with paths to training arrow files.')
@click.option('-k', '--num-anchors', type=int, default=16, show_default=True,
              help='Total number of anchors in the final table.')
@click.option('--per-shard/--global', default=True, show_default=True,
              help='Two-stage clustering. With --per-shard each shard first '
                   'contributes --per-shard-prototypes mini-anchors, then '
                   'the pooled mini-anchors are re-clustered to '
                   '--num-anchors. This gives every shard equal voice in '
                   'the final budget regardless of corpus mass, so the '
                   'curved tail (RTL, manuscripts) survives. With --global '
                   'all curves are pooled and clustered once.')
@click.option('--per-shard-prototypes', type=int, default=8, show_default=True,
              help='Mini-anchors per shard in stage 1 of --per-shard. Total '
                   'pool size is num_shards * this; the second stage '
                   're-clusters to --num-anchors.')
@click.option('-o', '--output', type=click.Path(dir_okay=False, writable=True),
              default='anchors.json', show_default=True,
              help='Where to write the new anchors.json.')
@click.option('--seed', type=int, default=0, show_default=True)
@click.argument('ground_truth', nargs=-1, callback=_expand_gt,
                type=click.Path(exists=False, dir_okay=False))
def cluster_anchors(training_data, num_anchors, per_shard, per_shard_prototypes,
                    output, seed, ground_truth):
    """
    Cluster baseline curves in chord-local space and write anchors.json.

    The output preserves the legacy on-disk layout (image-space cubic
    Bezier control points), so the model continues to load it via the
    existing ``CurveRegressionHead`` constructor; the head converts to
    chord-local form per forward pass.

    --per-shard (default) runs a two-stage clustering: each arrow file
    contributes a fixed number of mini-anchors regardless of its size,
    then the pooled mini-anchors are re-clustered to the final
    --num-anchors budget. This decouples a shard's voice from its corpus
    mass: a 500-line Yiddish shard contributes the same eight templates
    as a 50k-line German shard, so curved prototypes from RTL and
    manuscript scripts are not drowned out in the final clustering.
    """
    import numpy as np
    import torch
    from orli.modules.bezier import bezier_to_chord_local, chord_local_to_bezier

    files = list(ground_truth)
    if training_data:
        files.extend(training_data)
    if not files:
        raise click.UsageError(
            'No training arrow files supplied. Pass them positionally or '
            'via -t/--training-files.')

    if per_shard:
        proto_pool = []
        skipped = 0
        for path in files:
            curves = _read_curves_from_arrow(str(path))
            if not curves:
                skipped += 1
                continue
            cl = bezier_to_chord_local(torch.tensor(curves, dtype=torch.float32)).numpy()
            if len(cl) <= per_shard_prototypes:
                # Small shard: pass through whatever we have as prototypes.
                proto_pool.append(cl)
                message(f'  {Path(path).name}: {len(cl)} lines -> {len(cl)} prototypes (passthrough)')
                continue
            try:
                protos = _cluster_chord_local(cl, per_shard_prototypes, seed=seed)
            except click.ClickException as exc:
                click.echo(f'Skipping {path}: {exc.message}', err=True)
                skipped += 1
                continue
            proto_pool.append(protos)
            message(f'  {Path(path).name}: {len(cl)} lines -> {per_shard_prototypes} prototypes')
        if not proto_pool:
            raise click.ClickException('No clusterable data was found.')
        prototypes = np.concatenate(proto_pool, axis=0).astype(np.float32)
        message(f'Stage 2: re-clustering {len(prototypes)} prototypes from '
                f'{len(files) - skipped} shards to {num_anchors} anchors.')
        centres = _cluster_chord_local(prototypes, num_anchors, seed=seed)
    else:
        all_curves = []
        for path in files:
            all_curves.extend(_read_curves_from_arrow(str(path)))
        if not all_curves:
            raise click.ClickException('No clusterable data was found.')
        cl = bezier_to_chord_local(torch.tensor(all_curves, dtype=torch.float32)).numpy()
        message(f'Total lines: {len(cl)}; clustering to {num_anchors} anchors.')
        centres = _cluster_chord_local(cl, num_anchors, seed=seed)

    centres_t = torch.tensor(centres, dtype=torch.float32)
    image_space = chord_local_to_bezier(centres_t).tolist()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fp:
        json.dump(image_space, fp, indent=2)
    message(f'Wrote {len(image_space)} anchors to {out_path}.')


if __name__ == '__main__':
    cluster_anchors()
