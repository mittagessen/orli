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
              default='32',
              type=click.Choice(['64', '32', 'bf16', '16']),
              help='Numerical precision to use for training. Default is 32-bit single-point precision.')
@click.option('-2', '--autocast', default=False, show_default=True, flag_value=True,
              help='On compatible devices, uses autocast for `segment` which lower the memory usage.')
def cli(ctx, verbose, seed, deterministic, device, precision, autocast):
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
    ctx.meta['autocast'] = autocast
    set_logger(logger, level=30 - min(10 * verbose, 20))

cli.add_command(train)

if __name__ == '__main__':
    cli()
