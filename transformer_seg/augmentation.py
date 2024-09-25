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
import logging

logger = logging.getLogger(__name__)


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
