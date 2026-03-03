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
orli.cli.test
~~~~~~~~~~~~~

Command line driver for model evaluation.
"""
import click


@click.command('test')
@click.pass_context
@click.option('-B', '--batch-size',
              type=int,
              show_default=True,
              help='Batch size for evaluation')
@click.option('-m',
              '--load',
              type=click.Path(exists=True, readable=True),
              help='Path to safetensors or checkpoint file containing orli weights')
@click.option('-t', '--tolerance',
              type=float,
              show_default=True,
              help='Tolerance in pixels for baseline matching')
@click.option('--match-threshold',
              type=float,
              show_default=True,
              help='Minimum match score for ordering evaluation')
@click.option('--compile/--no-compile',
              help='Switch to enable/disable torch.compile() on model',
              default=True,
              show_default=True)
@click.option('--quantize/--no-quantize',
              help='Switch to enable/disable PTQ',
              default=False,
              show_default=True)
@click.argument('test_data', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def test(ctx, **kwargs):
    """
    Evaluates a model on an Arrow test dataset.

    Computes baseline detection metrics (Precision, Recall, F1) and
    reading order metrics (Spearman footrule, Kendall tau).
    """
    params = {}
    if ctx.default_map:
        params.update(ctx.default_map)
    params.update(ctx.params)
    params.update(ctx.meta)

    load = params.pop('load', None)
    test_data = params.pop('test_data', [])

    import torch

    from lightning.pytorch import Trainer
    from threadpoolctl import threadpool_limits
    from rich.table import Table
    from rich.console import Console
    from lightning.pytorch.callbacks import RichProgressBar

    from orli.configs import (OrliSegmentationTrainingConfig,
                              OrliSegmentationTrainingDataConfig,
                              OrliSegmentationTestConfig)
    from orli.model import OrliSegmentationDataModule, OrliSegmentationModel

    if load is None:
        raise click.UsageError('No model weights were provided. Pass a checkpoint or safetensors file.')

    if not test_data:
        raise click.UsageError('No test data was provided. Pass one or more Arrow files.')

    if params.get('quantize'):
        params['precision'] = 'bf16-true'

    accelerator = ctx.meta['accelerator']
    devices = ctx.meta['devices']

    dm_config = OrliSegmentationTrainingDataConfig(test_data=list(test_data),
                                                   **params)
    m_config = OrliSegmentationTrainingConfig(**params)

    data_module = OrliSegmentationDataModule(dm_config)

    load = str(load)
    if load.endswith('.ckpt'):
        model = OrliSegmentationModel.load_from_checkpoint(load, config=m_config)
    else:
        model = OrliSegmentationModel.load_from_weights(load, config=m_config)

    if params.get('compile'):
        click.echo('Compiling model ', nl=False)
        try:
            model.net = torch.compile(model.net, mode='max-autotune')
            click.secho('\u2713', fg='green')
        except Exception:
            click.secho('\u2717', fg='red')

    if params.get('quantize'):
        click.echo('Quantizing model ', nl=False)
        try:
            import torchao
            torchao.quantization.utils.recommended_inductor_config_setter()
            click.secho('\u2713', fg='green')
        except Exception:
            click.secho('\u2717', fg='red')

    test_config = OrliSegmentationTestConfig(**params)

    model.configure_test(test_config)

    cbs = [RichProgressBar(leave=True)]

    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      precision=params['precision'],
                      enable_progress_bar=True if not ctx.meta['verbose'] else False,
                      deterministic=params['deterministic'],
                      enable_model_summary=False,
                      callbacks=cbs,
                      num_sanity_val_steps=0)

    with threadpool_limits(limits=params['num_threads']):
        trainer.test(model, datamodule=data_module)

    results = model.test_results
    if not results:
        click.secho('No pages evaluated.', fg='red')
        return

    console = Console()
    table = Table(title='Evaluation Results')
    table.add_column('Metric', style='bold')
    table.add_column('Value', justify='right')

    table.add_row('Pages evaluated', str(results['num_pages']))
    table.add_row('Avg predicted lines/page', f'{results["avg_num_pred"]:.1f}')
    table.add_row('Avg GT lines/page', f'{results["avg_num_gt"]:.1f}')
    table.add_row('', '')
    table.add_row('Detection Precision', f'{results["precision"]:.4f}')
    table.add_row('Detection Recall', f'{results["recall"]:.4f}')
    table.add_row('Detection F1', f'{results["f1"]:.4f}')
    table.add_row('', '')
    footrule = results['spearman_footrule']
    tau = results['kendall_tau']
    table.add_row('Spearman Footrule (norm)',
                  f'{footrule:.4f}' if footrule == footrule else 'N/A')
    table.add_row('Kendall Tau',
                  f'{tau:.4f}' if tau == tau else 'N/A')

    console.print(table)
