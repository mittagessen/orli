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
import torch
import logging

from dataclasses import asdict

from kraken.containers import Segmentation, BaselineLine

from typing import TYPE_CHECKING, Union, Tuple, Optional, Literal, Generator

from orli.dataset import get_default_transforms


logging.captureWarnings(True)
logger = logging.getLogger('orli')


if TYPE_CHECKING:
    from orli.fusion import OrliModel
    from PIL import Image
    from lightning.fabric import Fabric

__all__ = ['batched_pred']


def sample_curves():
    pass


def segment(model: 'OrliModel',
            im: 'Image.Image',
            fabric: 'Fabric') -> Segmentation
    """
    Baseline segmentation

    Args:
        model: OrliModel for generation.
        im: Pillow image
        bounds: Segmentation for input image
        fabric: Fabric context manager to cast models and tensors.
        prompt_mode: How to embed line positional prompts. Per default prompts
                     are determined by the segmentation type if the model
                     indicates either curves or boxes are supported. If the
                     model supports only boxes and the input segmentation is
                     baseline-type, bounding boxes will be generated from the
                     bounding polygon if available. If the model expects curves
                     and the segmentation is of bounding box-type an exception
                     will be raised. If explicit values are set.
        batch_size: Number of lines to predict in parallel

    Yields:
        An ocr_record containing the recognized text, dummy character
        positions, and confidence values for each character.

    Raises:
        ValueError when the model expects curves and the segmentation of bounding box-type.
    """
    m_dtype = next(model.parameters()).dtype
    m_device = next(model.parameters()).device

    # load image transforms
    im_transforms = get_default_transforms(dtype=m_dtype)

    # prepare model for generation
    model.prepare_for_generation(batch_size=1, device=m_device)
    model = model.eval()

    with fabric.init_tensor(), torch.inference_mode():
        image_input = im_transforms(im).unsqueeze(0).to(m_device)
        curves = model.predict(encoder_input=image_input)
