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
orli.cli.compile
~~~~~~~~~~~~~~~~

Command line driver for binary dataset compilation.
"""
import click
import logging

from .util import _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('orli')


@click.command('compile')
@click.pass_context
@click.option('-o', '--output', type=click.Path(), default='dataset.arrow',
              help='Output dataset file')
@click.option('-F', '--files', default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data.')
@click.option('-r', '--resize', nargs=2, type=int, default=None,
              help='Resize images to fixed (height, width)')
@click.option('--allow-textless/--no-allow-textless', default=False,
              help='Include lines without text in the dataset')
@click.argument('ground_truth', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def compile(ctx, **params):
    """
    Precompiles a binary baseline dataset from a collection of XML files.
    """
    params = ctx.params.copy()
    params.update(ctx.meta)

    files = params.pop('files', [])
    ground_truth = list(params.pop('ground_truth', []))

    if files:
        ground_truth.extend(files)

    if not ground_truth:
        raise click.UsageError('No training data was provided to the compile command. Use the `ground_truth` argument or `-F`.')

    from rich.progress import MofNCompleteColumn, Progress, TimeElapsedColumn

    from orli import dataset

    with Progress(*Progress.get_default_columns(),
                  TimeElapsedColumn(),
                  MofNCompleteColumn()) as progress:
        extract_task = progress.add_task('Compiling dataset',
                                         total=0,
                                         start=False,
                                         visible=not bool(ctx.meta.get('verbose')))

        def _update_bar(advance, total):
            if not progress.tasks[0].started:
                progress.start_task(extract_task)
            progress.update(extract_task, total=total, advance=advance)

        stats = dataset.compile(ground_truth,
                                params['output'],
                                resize=tuple(params['resize']) if params['resize'] else None,
                                allow_textless=params['allow_textless'],
                                callback=_update_bar)

    message(f'Output file written to {params["output"]}')
    message(f'Pages: {stats["num_pages"]}')
    message(f'Lines: {stats["num_lines"]}')
    message(f'Maximum lines in a page: {stats["max_lines_in_page"]}')
