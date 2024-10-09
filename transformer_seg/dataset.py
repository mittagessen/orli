#
# Copyright 2015 Benjamin Kiessling
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
import io
import torch
import lightning.pytorch as L

import pyarrow as pa

from typing import (TYPE_CHECKING, Callable, List, Literal, Optional, Union,
                    Sequence)

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence

from transformers import DonutImageProcessor

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['LineSegmentationDataModule']

import logging

logger = logging.getLogger(__name__)


def collate_curves(batch):
    """
    Concatenates and pads curves.
    """
    return {'image': default_collate([item['image'] for item in batch]),
            'target': pad_sequence([item['target'] for item in batch], batch_first=True, padding_value=-100)}


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


class LineSegmentationDataModule(L.LightningDataModule):
    def __init__(self,
                 training_data: Union[str, 'PathLike'],
                 evaluation_data: Union[str, 'PathLike'],
                 curve_resolution: int = 1000,
                 eos_token_id: int = 2,
                 augmentation: bool = False,
                 batch_size: int = 1,
                 num_workers: int = 8):
        super().__init__()

        self.save_hyperparameters()
        self.im_transforms = DonutImageProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

    def setup(self, stage: str):
        """
        Actually builds the datasets.
        """
        self.train_set = BaselineSegmentationDataset(self.hparams.training_data,
                                                     im_transforms=self.im_transforms,
                                                     augmentation=self.hparams.augmentation,
                                                     eos_token_id=self.hparams.eos_token_id,
                                                     curve_resolution=self.hparams.curve_resolution)
        self.val_set = BaselineSegmentationDataset(self.hparams.evaluation_data,
                                                   im_transforms=self.im_transforms,
                                                   eos_token_id=self.hparams.eos_token_id,
                                                   curve_resolution=self.hparams.curve_resolution,
                                                   augmentation=False)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_curves)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=collate_curves)


class BaselineSegmentationDataset(Dataset):
    """
    Dataset for training a line segmentation model from baseline data.

    Args:
        files: Paths to binary dataset files.
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        augmentation: Enables augmentation.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms=None,
                 augmentation: bool = False,
                 curve_resolution: int = 1000,
                 eos_token_id: int = 2) -> None:
        self.files = files
        self.transforms = im_transforms
        self.aug = None
        self.arrow_table = None
        self.curve_resolution = curve_resolution
        self.eos_token_id = eos_token_id

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                try:
                    ds_table = pa.ipc.open_file(source).read_all()
                except Exception:
                    logger.warning(f'{file} is not an arrow file')
                    continue
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])

        if augmentation:
            from transformer_seg.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __getitem__(self, index: int):
        # just sample from a random page
        item = self.arrow_table.column('pages')[index].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        im = Image.open(io.BytesIO(im))
        im = torch.tensor(self.transforms(im)['pixel_values'][0])

        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.from_numpy(o['image'].transpose(2, 0, 1))

        lines = (torch.tensor([x['curve'] for x in page_data]) * self.curve_resolution).to(torch.long)
        # offset behind sos/eos/pad token indices
        lines += 3
        # append eos token
        lines = torch.clamp(torch.cat([lines, torch.full((1, 8), self.eos_token_id)]), min=0, max=self.curve_resolution+3)
        return {'image': im, 'target': lines}

    def __len__(self) -> int:
        return len(self.arrow_table)
