#! /usr/bin/env python
"""
Computes representative curve anchors from one or more source datasets.

The default strategy is coverage-oriented instead of density-oriented:
curves are transformed to logit space, robustly scaled, and a greedy k-center
selection is run on the transformed representation. This yields anchors that
cover the support of the dataset better than plain k-means, which tends to
collapse onto the most frequent modes.
"""
import click
import numpy as np

from rich.progress import track


def _inverse_sigmoid(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _robust_scale(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(x, axis=0)
    mad = np.median(np.abs(x - center), axis=0)
    scale = np.maximum(1.4826 * mad, eps)
    return (x - center) / scale, center, scale


def _sq_dist_to_point(points: np.ndarray, point: np.ndarray) -> np.ndarray:
    diff = points - point
    return np.einsum('ij,ij->i', diff, diff)


def _greedy_kcenter(points: np.ndarray, num_anchors: int) -> np.ndarray:
    """
    Greedy k-center / farthest-point traversal in Euclidean space.

    The first anchor is chosen as the point nearest the robust center (the
    origin after scaling); subsequent anchors maximize the minimum distance to
    any previously selected anchor.
    """
    if num_anchors >= len(points):
        return np.arange(len(points), dtype=np.int64)

    selected = np.empty((num_anchors,), dtype=np.int64)
    selected[0] = int(np.argmin(np.einsum('ij,ij->i', points, points)))

    min_sq_dist = _sq_dist_to_point(points, points[selected[0]])

    for idx in range(1, num_anchors):
        selected[idx] = int(np.argmax(min_sq_dist))
        candidate_dist = _sq_dist_to_point(points, points[selected[idx]])
        np.minimum(min_sq_dist, candidate_dist, out=min_sq_dist)

    return selected


def _coverage_anchors(lines: np.ndarray, num_anchors: int) -> np.ndarray:
    transformed = _inverse_sigmoid(lines)
    transformed, _, _ = _robust_scale(transformed)
    selected = _greedy_kcenter(transformed, num_anchors)
    return lines[selected]


@click.command()
@click.option('-n', '--num-anchors', default=5, help='Number of anchors to compute.', show_default=True)
@click.option('--method',
              type=click.Choice(['coverage', 'kmeans']),
              default='coverage',
              show_default=True,
              help='Anchor selection strategy.')
@click.option('-o', '--output',
              default='anchors.json',
              show_default=True,
              type=click.Path(dir_okay=False, writable=True),
              help='Output file for anchors.')
@click.argument('files', nargs=-1)
def cli(num_anchors, method, output, files):
    """
    Computes `n` anchors from one or more binary dataset files and writes them
    to a JSON file.
    """
    if not files:
        raise click.UsageError('No dataset files given.')
    import json
    import pyarrow as pa

    arrow_table = None
    for file in track(files, description="Loading tables..."):
        with pa.memory_map(file, 'rb') as source:
            ds_table = pa.ipc.open_file(source).read_all()
            if not arrow_table:
                arrow_table = ds_table
            else:
                arrow_table = pa.concat_tables([arrow_table, ds_table])

    lines = []
    for item in track(arrow_table.column('pages'), description="Reading lines..."):
        item = item.as_py()
        page_data = item['lines']
        for line in page_data:
            lines.append(line['curve'])
    if not lines:
        raise click.UsageError('No lines found in the provided datasets.')

    lines = np.asarray(lines, dtype=np.float32)

    if method == 'coverage':
        selected = _coverage_anchors(lines, num_anchors)
        anchors = [tuple(float(v) for v in row) for row in selected]
    else:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=num_anchors).fit(lines)
        anchors = [tuple(float(v) for v in row) for row in kmeans.cluster_centers_]

    print(f'Anchors: {anchors}')
    with open(output, 'w') as fp:
        json.dump(anchors, fp, indent=2)


if __name__ == '__main__':
    cli()
