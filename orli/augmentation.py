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
Utility functions for data loading and training of VGSL networks.
"""
from typing import Any, Dict, List, Tuple, Union

import torch
import logging

from torchvision.transforms import v2

logger = logging.getLogger(__name__)


class BoundRandomResize(v2.RandomShortestSize):
    """
    Randomly proportionally resizes an input image between upper and lower
    image dimension bounds.

    Only accepts tuple-d bounds.
    """
    def __init__(self,
                 min_size: Union[List[int], Tuple[int]],
                 max_size: Union[List[int], Tuple[int]],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs, min_size=min_size)
        self.min_size = list(min_size)
        self.max_size = list(max_size)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        orig_height, orig_width = v2._utils.query_size(flat_inputs)

        # recompute lower and upper ratios so resized image falls into bounds
        min_r = max(self.min_size[0]/orig_height, self.min_size[1]/orig_width)
        max_r = min(self.max_size[0]/orig_height, self.max_size[1]/orig_width)

        r = torch.FloatTensor(1).uniform_(min_r, max_r).item()
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        return dict(size=(new_height, new_width))


class DefaultAugmenter:
    def __init__(self):
        import cv2
        cv2.setNumThreads(0)
        from albumentations import (Blur, Compose, MedianBlur, MotionBlur,
                                    OneOf, PixelDropout, ToFloat, ColorJitter)

        self._transforms = Compose([
                                    ToFloat(),
                                    PixelDropout(p=0.2),
                                    ColorJitter(p=0.5),
                                    OneOf([
                                        MotionBlur(p=0.2),
                                        MedianBlur(blur_limit=3, p=0.1),
                                        Blur(blur_limit=3, p=0.1),
                                    ], p=0.2)
                                   ], p=0.5)

    def __call__(self, image):
        return self._transforms(image=image)
