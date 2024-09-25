#
# Copyright 2024 Benjamin Kiessling
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
party.cli.test
~~~~~~~~~~~~~~~~~~

Command line driver for recognition tests.
"""
import logging

import click
from threadpoolctl import threadpool_limits
from typing import List

from party.default_specs import RECOGNITION_HYPER_PARAMS

from .util import _expand_gt, _validate_manifests, message

logging.captureWarnings(True)
logger = logging.getLogger('party')

# suppress worker seeding message
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)


@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size', show_default=True, type=click.INT,
              default=RECOGNITION_HYPER_PARAMS['batch_size'], help='Batch sample size')
@click.option('-m', '--model', show_default=True, type=click.Path(exists=True, readable=True),
              help='Model to evaluate')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data.')
@click.option('-d', '--device', show_default=True, default='cpu', help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--pad', show_default=True, type=click.INT, default=16, help='Left and right '
              'padding around lines')
@click.option('--workers', show_default=True, default=1,
              type=click.IntRange(0),
              help='Number of worker processes when running on CPU.')
@click.option('--threads', show_default=True, default=1,
              type=click.IntRange(1),
              help='Max size of thread pools for OpenMP/BLAS operations.')
@click.option('--reorder/--no-reorder', show_default=True, default=True, help='Reordering of code points to display order')
@click.option('--base-dir', show_default=True, default='auto',
              type=click.Choice(['L', 'R', 'auto']), help='Set base text '
              'direction.  This should be set to the direction used during the '
              'creation of the training data. If set to `auto` it will be '
              'overridden by any explicit value given in the input files.')
@click.option('-u', '--normalization', show_default=True, type=click.Choice(['NFD', 'NFKD', 'NFC', 'NFKC']),
              default=None, help='Ground truth normalization')
@click.option('-n', '--normalize-whitespace/--no-normalize-whitespace',
              show_default=True, default=True, help='Normalizes unicode whitespace')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with JSON `.path` files '
              'containing the baseline information. In `binary` mode files are '
              'collections of pre-extracted text line images.')
@click.option('--fixed-splits/--ignore-fixed-split', show_default=True, default=False,
              help='Whether to honor fixed splits in binary datasets.')
@click.argument('test_set', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def test(ctx, batch_size, model, evaluation_files, device, pad, workers,
         threads, reorder, base_dir, normalization, normalize_whitespace,
         format_type, fixed_splits, test_set):
    """
    Evaluate on a test set.
    """
    if not model:
        raise click.UsageError('No model to evaluate given.')

    import torch
    import numpy as np

    from torch.utils.data import DataLoader

    from torchmetrics.text import CharErrorRate, WordErrorRate

    from kraken.lib import util
    from kraken.lib.exceptions import KrakenInputException
    from kraken.lib.dataset import (ArrowIPCRecognitionDataset,
                                    GroundTruthDataset, ImageInputTransforms,
                                    PolygonGTDataset, compute_confusions,
                                    global_align, collate_sequences)
    from kraken.lib.progress import KrakenProgressBar
    from kraken.lib.xml import XMLPage
    from kraken.serialization import render_report

    from party.pred import PytorchRecognitionModel

    logger.info('Building test set from {} line images'.format(len(test_set) + len(evaluation_files)))

    message('Loading model {}\t'.format(model), nl=False)
    nn = PytorchRecognitionModel.load_checkpoint(model).to(device)
    message('\u2713', fg='green')

    dataset_kwargs = {}

    pin_ds_mem = False
    if device != 'cpu':
        pin_ds_mem = True

    test_set = list(test_set)

    if evaluation_files:
        test_set.extend(evaluation_files)

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if fixed_splits:
        if format_type != "binary":
            logger.warning("--fixed-splits can only be use with data using binary format")
        else:
            dataset_kwargs["split_filter"] = "test"

    if format_type in ['xml', 'page', 'alto']:
        test_set = [{'page': XMLPage(file, filetype=format_type).to_container()} for file in test_set]
        DatasetClass = PolygonGTDataset
    elif format_type == 'binary':
        DatasetClass = ArrowIPCRecognitionDataset
        test_set = [{'file': file} for file in test_set]
    else:
        DatasetClass = GroundTruthDataset
        test_set = [{'line': util.parse_gt_path(img)} for img in test_set]

    if len(test_set) == 0:
        raise click.UsageError('No evaluation data was provided to the test command. Use `-e` or the `test_set` argument.')

    if reorder and base_dir != 'auto':
        reorder = base_dir

    cer_list = []
    wer_list = []

    with threadpool_limits(limits=threads):
        algn_gt: List[str] = []
        algn_pred: List[str] = []
        chars = 0
        error = 0
        message('Evaluating {}'.format(model))
        logger.info('Evaluating {}'.format(model))
        ts = ImageInputTransforms(batch_size, nn.height, 0, 1, (pad, 0), False)
        ds = DatasetClass(normalization=normalization,
                          whitespace_normalization=normalize_whitespace,
                          reorder=reorder,
                          im_transforms=ts,
                          augmentation=False,
                          **dataset_kwargs)
        for line in test_set:
            try:
                ds.add(**line)
            except ValueError as e:
                raise
                logger.info(e)

        # don't encode validation set as the alphabets may not match causing encoding failures
        ds.no_encode()
        ds_loader = DataLoader(ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               pin_memory=pin_ds_mem,
                               collate_fn=collate_sequences)

        test_cer = CharErrorRate()
        test_wer = WordErrorRate()

        with KrakenProgressBar() as progress:
            batches = len(ds_loader)
            pred_task = progress.add_task('Evaluating', total=batches, visible=True if not ctx.meta['verbose'] else False)

            for batch in ds_loader:
                im = batch['image']
                text = batch['target']
                lens = batch['seq_lens']
                im = im.to(device)
                try:
                    pred = nn.predict_string(im, lens)
                    for x, y in zip(pred, text):
                        chars += len(y)
                        c, algn1, algn2 = global_align(y, x)
                        algn_gt.extend(algn1)
                        algn_pred.extend(algn2)
                        error += c
                        test_cer.update(x, y)
                        test_wer.update(x, y)

                except FileNotFoundError as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning('{} {}. Skipping.'.format(e.strerror, e.filename))
                except KrakenInputException as e:
                    batches -= 1
                    progress.update(pred_task, total=batches)
                    logger.warning(str(e))
                progress.update(pred_task, advance=1)

        cer_list.append(1.0 - test_cer.compute())
        wer_list.append(1.0 - test_wer.compute())
        confusions, scripts, ins, dels, subs = compute_confusions(algn_gt, algn_pred)
        rep = render_report(model,
                            chars,
                            error,
                            cer_list[-1],
                            wer_list[-1],
                            confusions,
                            scripts,
                            ins,
                            dels,
                            subs)
        logger.info(rep)
        message(rep)

    logger.info('Average character accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(cer_list) * 100, np.std(cer_list) * 100))
    message('Average character accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(cer_list) * 100, np.std(cer_list) * 100))
    logger.info('Average word accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(wer_list) * 100, np.std(wer_list) * 100))
    message('Average word accuracy: {:0.2f}%, (stddev: {:0.2f})'.format(np.mean(wer_list) * 100, np.std(wer_list) * 100))
