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
orli.pred
~~~~~~~~~

API for inference
"""
import uuid
import torch
import logging
import numpy as np

from dataclasses import asdict

from scipy.special import comb
from shapely.geometry import LineString
from kraken.containers import Segmentation, BaselineLine

from typing import TYPE_CHECKING, Union, Tuple, Optional, Literal, Generator, List

from orli.dataset import get_default_transforms


logging.captureWarnings(True)
logger = logging.getLogger('orli')


if TYPE_CHECKING:
    from orli.fusion import OrliModel
    from PIL import Image
    from lightning.fabric import Fabric

__all__ = ['segment']

# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n, k)


def BezierCoeff(ts):
    return [[Mtk(3, t, k) for k in range(4)] for t in ts]


def sample_curves(curves: torch.Tensor) -> List[List[Tuple[int, int]]]:
    samples = np.linspace(0, 1, 20)
    coeff = np.array(BezierCoeff(samples))
    lines = []
    for curve in curves:
        curve = np.array(curve.cpu())
        curve.resize(4, 2)
        lines.append(coeff.dot(curve))
    return lines


def segment(model: 'OrliModel',
            im: 'Image.Image',
            fabric: 'Fabric') -> Segmentation:
    """
    Baseline segmentation

    Args:
        model: OrliModel for generation.
        im: Pillow image
        fabric: Fabric context manager to cast models and tensors.

    Returns: 
        A Segmentation object 
    """
    m_dtype = next(model.parameters()).dtype
    m_device = next(model.parameters()).device

    curve_scale = torch.tensor((im.width, im.height) * 4)

    # load image transforms
    im_transforms = get_default_transforms(dtype=m_dtype)

    # prepare model for generation
    model.prepare_for_generation(batch_size=1, device=m_device)
    model = model.eval()

    with fabric.init_tensor(), torch.inference_mode():
        image_input = im_transforms(im).unsqueeze(0).to(m_device)
        curves = model.predict(encoder_input=image_input).squeeze()
        # strip trailing no-op if it is there.
        if not curves[-1].any():
           curves = curves[:-1]
        curves *= curve_scale
    lines = sample_curves(curves)

    line_containers = []
    for line in lines:
        boundary = LineString(line).buffer(-10, single_sided=True)
        boundary = np.array(boundary.boundary.coords)
        boundary = np.rint(boundary).astype('int')
        boundary = tuple(map(tuple, boundary.tolist()))
        line = np.rint(line).astype('int')
        baseline = tuple(map(tuple, line.tolist()))
        line_containers.append(BaselineLine(id=f'_{uuid.uuid4()}',
                                            baseline=baseline,
                                            boundary=boundary,
                                            tags={'type': [{'type': 'default'}]}))
    return Segmentation(type='baselines',
                        imagename=im.filename,
                        script_detection=False,
                        text_direction='horizontal-lr',
                        lines=line_containers)
