#
# Copyright 2022 Benjamin Kiessling
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
orli.cli.pred
~~~~~~~~~~~~~

Command line driver for baseline detection
"""
import uuid
import click
import logging

from pathlib import Path

from .util import to_ptl_device, get_input_parser

logging.captureWarnings(True)
logger = logging.getLogger('party')


@click.command('segment')
@click.pass_context
@click.option('-i', '--input',
              type=(click.Path(exists=True, dir_okay=False, path_type=Path),  # type: ignore
                    click.Path(writable=True, dir_okay=False, path_type=Path)),
              multiple=True,
              help='Input-output file pairs. Each input file (first argument) is mapped to one '
                   'output file (second argument), e.g. `-i input.jpg output.alto`')
@click.option('-I', '--batch-input', multiple=True, help='Glob expression to add multiple files at once.')
@click.option('-o', '--suffix', default='', show_default=True,
              help='Suffix for output files from batch and PDF inputs.')
@click.option('-m', '--load-from-repo',
              default=None,
              show_default=True,
              help="HTRMoPo identifier of the party model to evaluate")
@click.option('-mi', '--load-from-file',
              default=None,
              show_default=True,
              help="Path to the party model to evaluate")
@click.option('-h', '--hocr', 'serializer',
              help='Switch between hOCR, ALTO, abbyyXML, PageXML or "native" '
              'output. Native are plain image files for image, JSON for '
              'segmentation, and text for transcription output.',
              flag_value='hocr')
@click.option('-a', '--alto', 'serializer', flag_value='alto')
@click.option('-y', '--abbyy', 'serializer', flag_value='abbyyxml')
@click.option('-x', '--pagexml', 'serializer', flag_value='pagexml')
@click.option('--compile/--no-compile', help='Switch to enable/disable torch.compile() on model', default=True, show_default=True)
@click.option('--quantize/--no-quantize', help='Switch to enable/disable PTQ', default=False, show_default=True)
@click.option('-b', '--batch-size', default=1, help='Set batch size in generator')
def segment(ctx, input, batch_input, suffix, load_from_repo, load_from_file,
            serializer, compile, quantize, batch_size):
    """
    Segments images and writes line segmentation output in ALTO XML format.
    """
    # try importing kraken as we need it for inference
    try:
        from kraken.containers import ProcessingStep
        from kraken.lib.progress import KrakenProgressBar, KrakenDownloadProgressBar

    except ImportError:
        raise click.UsageError('Inference requires the kraken package')

    if load_from_file and load_from_repo:
        raise click.BadOptionUsage('load_from_file', 'load_from_* options are mutually exclusive.')
    elif load_from_file is None and load_from_repo is None:
        load_from_repo = '10.5281/zenodo.14616981'

    import os
    import glob
    import torch

    from PIL import Image
    from pathlib import Path
    from lightning.fabric import Fabric

    from htrmopo import get_model

    from threadpoolctl import threadpool_limits

    from orli.pred import segment as segment_
    from orli.fusion import OrliModel

    try:
        accelerator, device = to_ptl_device(ctx.meta['device'])
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    if load_from_repo:
        with KrakenDownloadProgressBar() as progress:
            download_task = progress.add_task(f'Downloading {load_from_repo}', total=0, visible=True)
            load_from_file = get_model(load_from_repo,
                                       callback=lambda total, advance: progress.update(download_task, total=total, advance=advance)) / 'model.safetensors'

    # parse input files
    input = list(input)
    # expand batch inputs
    if batch_input and suffix:
        for batch_expr in batch_input:
            for in_file in glob.glob(os.path.expanduser(batch_expr), recursive=True):
                input.append((in_file, '{}{}'.format(os.path.splitext(in_file)[0], suffix)))

    # torchao expects bf16 weights
    if quantize:
        ctx.meta['precision'] = 'bf16-true'

    fabric = Fabric(accelerator=accelerator,
                    devices=device,
                    precision=ctx.meta['precision'])

    steps = [ProcessingStep(id=str(uuid.uuid4()),
                            category='processing',
                            description='Baseline segmentation',
                            settings={})]

    with torch.inference_mode(), threadpool_limits(limits=ctx.meta['threads']), fabric.init_tensor(), fabric.init_module():

        model = OrliModel.from_safetensors(load_from_file)

        if compile:
            click.echo('Compiling model ', nl=False)
            try:
                model = torch.compile(model, mode='max-autotune')
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

        with KrakenProgressBar() as progress:
            file_prog = progress.add_task('Files', total=len(input))
            for input_file, output_file in input:
                input_file = Path(input_file)
                output_file = Path(output_file)
                imagename = get_input_parser(ctx.meta['input_format_type'])(input).imagename

                im = Image.open(imagename)
                res = segment_(model=model, im=im, fabric=fabric)
                with click.open_file(output_file, 'w', encoding='utf-8') as fp:
                    logger.info(f'Serializing as {serializer} into {output_file}')
                    from kraken import serialization
                    fp.write(serialization.serialize(res,
                                                     image_size=im.size,
                                                     template=serializer,
                                                     template_source='native',
                                                     processing_steps=steps))
                progress.update(file_prog, advance=1)
