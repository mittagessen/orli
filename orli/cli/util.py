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
orli.cli.util
~~~~~~~~~~~~~

Command line driver helpers
"""
import glob
import logging
import os
from typing import List, Optional, Tuple

import click
import lightning as L

from lightning.pytorch.callbacks import BaseFinetuning


logging.captureWarnings(True)
logger = logging.getLogger('orli')


def get_input_parser(type_str: str) -> Callable[[str], Dict[str, Any]]:
    if type_str in ['alto', 'page', 'xml']:
        from kraken.lib.xml import XMLPage
        return XMLPage
    elif type_str == 'image':
        return Image.open


def _validate_manifests(ctx, param, value):
    images = []
    for manifest in value:
        for entry in manifest.readlines():
            im_p = entry.rstrip('\r\n')
            if os.path.isfile(im_p):
                images.append(im_p)
            else:
                logger.warning('Invalid entry "{}" in {}'.format(im_p, manifest.name))
    return images


def _expand_gt(ctx, param, value):
    images = []
    for expression in value:
        images.extend([x for x in glob.iglob(expression, recursive=True) if os.path.isfile(x)])
    return images


def message(msg, **styles):
    if logger.getEffectiveLevel() >= 30:
        click.secho(msg, **styles)


def to_ptl_device(device: str) -> Tuple[str, Optional[List[int]]]:
    if device.strip() == 'auto':
        return 'auto', 'auto'
    devices = device.split(',')
    if devices[0] in ['cpu', 'mps']:
        return devices[0], 'auto'
    elif any([devices[0].startswith(x) for x in ['tpu', 'cuda', 'hpu', 'ipu']]):
        devices = [device.split(':') for device in devices]
        devices = [(x[0].strip(), x[1].strip()) for x in devices]
        if len(set(x[0] for x in devices)) > 1:
            raise Exception('Can only use a single type of device at a time.')
        dev, _ = devices[0]
        if dev == 'cuda':
            dev = 'gpu'
        return dev, [int(x[1]) for x in devices]
    raise Exception(f'Invalid device {device} specified')
