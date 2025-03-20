#! /usr/bin/env python
import logging

import click
from PIL import Image
from rich.logging import RichHandler
from rich.traceback import install

from .train import train


def set_logger(logger=None, level=logging.ERROR):
    logger.addHandler(RichHandler(rich_tracebacks=True))
    logger.setLevel(level)


# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2

logging.captureWarnings(True)
logger = logging.getLogger()

APP_NAME = 'tseg'

logging.captureWarnings(True)
logger = logging.getLogger(APP_NAME)

# install rich traceback handler
install(suppress=[click])

# raise default max image size to 20k * 20k pixels
Image.MAX_IMAGE_PIXELS = 20000 ** 2


@click.group()
@click.version_option()
@click.pass_context
@click.option('-v', '--verbose', default=0, count=True)
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-r', '--deterministic/--no-deterministic', default=False,
              help="Enables deterministic training. If no seed is given and enabled the seed will be set to 42.")
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('--precision',
              show_default=True,
              default='32-true',
              type=click.Choice(['transformer-engine', 'transformer-engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true']),
              help='Numerical precision to use for training. Default is 32-bit single-point precision.')
@click.option('--threads', default=1, show_default=True, type=click.IntRange(1),
              help='Size of thread pools for intra-op parallelization')
def cli(ctx, verbose, seed, deterministic, device, precision, threads):
    ctx.meta['deterministic'] = False if not deterministic else 'warn'
    if seed:
        from lightning.pytorch import seed_everything
        seed_everything(seed, workers=True)
    elif deterministic:
        from lightning.pytorch import seed_everything
        seed_everything(42, workers=True)

    ctx.meta['verbose'] = verbose
    ctx.meta['device'] = device
    ctx.meta['precision'] = precision
    ctx.meta['threads'] = threads
    set_logger(logger, level=30 - min(10 * verbose, 20))


cli.add_command(train)

if __name__ == '__main__':
    cli()
