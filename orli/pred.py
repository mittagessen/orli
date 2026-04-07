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
import logging

from typing import Union, Optional, TYPE_CHECKING

from kraken.tasks.segmentation import SegmentationTaskModel
from kraken.models import load_models
from orli.configs import OrliSegmentationInferenceConfig

if TYPE_CHECKING:
    from os import PathLike
    from PIL import Image

from kraken.containers import Segmentation

logging.captureWarnings(True)
logger = logging.getLogger('orli')

__all__ = ['segment']


def segment(im: 'Image.Image',
            model_path: Union[str, 'PathLike'],
            config: Optional[OrliSegmentationInferenceConfig] = None) -> Segmentation:
    """
    Performs baseline segmentation on an image.

    The interface is compatible with kraken's SegmentationTaskModel: models
    are loaded via ``kraken.models.load_models`` and wrapped in a
    ``SegmentationTaskModel`` so the standard ``prepare_for_inference`` /
    ``predict`` protocol is used.

    Args:
        im: PIL Image to segment.
        model_path: Path to a safetensors file containing orli weights.
        config: Inference configuration. If ``None`` a default
                :class:`OrliSegmentationInferenceConfig` is created.

    Returns:
        A :class:`Segmentation` container with detected baselines.
    """
    if config is None:
        config = OrliSegmentationInferenceConfig()
    models = load_models(model_path, tasks=['segmentation'])
    task = SegmentationTaskModel(models)
    return task.predict(im, config)
