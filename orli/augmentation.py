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
import torch
import logging

import torch.nn as nn

from kornia import augmentation as aug

logger = logging.getLogger(__name__)


class DefaultAugmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.transforms = nn.Sequential(
            aug.RandomErasing(p=0.2, scale=(0.02, 0.02)),
            aug.ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            aug.RandomGrayscale(p=0.2),
            aug.RandomChoice(
                [
                    aug.RandomMotionBlur(p=0.2, kernel_size=3, angle=45.0, direction=0.0),
                    aug.RandomMedianBlur(p=0.1, kernel_size=3),
                    aug.RandomGaussianBlur(p=0.1, kernel_size=(3, 3), sigma=(0.1, 2.0)),
                ],
                p=0.2,
            ),
        )

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transforms(image)
