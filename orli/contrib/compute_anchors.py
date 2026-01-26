#! /usr/bin/env python
"""
Computes k-means clustered anchors from a source dataset.
"""
import click
from rich.progress import track


@click.command()
@click.option('-n', '--num-anchors', default=5, help='Number of anchors to compute.', show_default=True)
@click.argument('files', nargs=-1)
def cli(num_anchors, files):
    """
    Computes `n` anchors from one or more binary dataset files and writes them
    to `anchors.pt`.
    """
    if not files:
        raise click.UsageError('No dataset files given.')
    import torch
    import numpy as np
    import pyarrow as pa

    from sklearn.cluster import KMeans

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
    lines = np.array(lines)
    kmeans = KMeans(n_clusters=num_anchors).fit(lines)
    anchors = torch.from_numpy(kmeans.cluster_centers_).float()
    print(f'Anchors: {anchors}')
    with open('anchors.pt', 'wb') as fp:
        torch.save(anchors, fp)


if __name__ == '__main__':
    cli()
