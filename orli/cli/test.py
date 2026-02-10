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
@click.option('-l', '--load',
              default=None,
              show_default=True,
              help='Path to safetensors or checkpoint file containing orli weights')
@click.option('-t', '--tolerance',
              type=float,
              default=10.0,
              show_default=True,
              help='Tolerance in pixels for baseline matching')
@click.option('--match-threshold',
              type=float,
              default=0.5,
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
def test(ctx, load, tolerance, match_threshold,
         compile, quantize, test_data):
    """
    Evaluates a model on an Arrow test dataset.

    Computes baseline detection metrics (Precision, Recall, F1) and
    reading order metrics (Spearman footrule, Kendall tau).
    """
    import torch

    from lightning.pytorch import Trainer
    from threadpoolctl import threadpool_limits
    from rich.table import Table
    from rich.console import Console

    from orli.configs import (OrliSegmentationTrainingConfig,
                              OrliSegmentationTrainingDataConfig,
                              OrliSegmentationTestConfig)
    from orli.model import OrliSegmentationDataModule, OrliSegmentationModel

    if load is None:
        raise click.UsageError('No model weights were provided. Pass a checkpoint or safetensors file.')

    if not test_data:
        raise click.UsageError('No test data was provided. Pass one or more Arrow files.')

    if quantize:
        ctx.meta['precision'] = 'bf16-true'

    accelerator = ctx.meta['accelerator']
    devices = ctx.meta['devices']

    dm_config = OrliSegmentationTrainingDataConfig(test_data=list(test_data),
                                                   num_workers=ctx.meta.get('num_workers'))
    data_module = OrliSegmentationDataModule(dm_config)
    model_config = OrliSegmentationTrainingConfig()

    load = str(load)
    if load.endswith('.ckpt'):
        model = OrliSegmentationModel.load_from_checkpoint(load, config=model_config)
    else:
        model = OrliSegmentationModel.load_from_weights(load, config=model_config)

    if compile:
        click.echo('Compiling model ', nl=False)
        try:
            model.net = torch.compile(model.net, mode='max-autotune')
            click.secho('\u2713', fg='green')
        except Exception:
            click.secho('\u2717', fg='red')

    if quantize:
        click.echo('Quantizing model ', nl=False)
        try:
            import torchao
            torchao.quantization.utils.recommended_inductor_config_setter()
            click.secho('\u2713', fg='green')
        except Exception:
            click.secho('\u2717', fg='red')

    test_config = OrliSegmentationTestConfig(device=devices,
                                             accelerator=accelerator,
                                             precision=ctx.meta.get('precision'),
                                             batch_size=dm_config.batch_size,
                                             tolerance=tolerance,
                                             match_threshold=match_threshold)

    model.configure_test(test_config)

    trainer = Trainer(accelerator=accelerator,
                      devices=devices,
                      precision=ctx.meta.get('precision'),
                      enable_progress_bar=False if ctx.meta.get('verbose') else True,
                      enable_model_summary=False,
                      logger=False,
                      num_sanity_val_steps=0)

    with threadpool_limits(limits=ctx.meta.get('num_threads')):
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
