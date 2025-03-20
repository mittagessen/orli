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
import torch.nn.functional as F
import lightning.pytorch as L

import numpy as np
import pyarrow as pa

from functools import partial
from torchvision.transforms import v2
from typing import TYPE_CHECKING, Union, Sequence

from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torch.utils.data import default_collate

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['LineSegmentationDataModule']

import logging

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 20000 ** 2


def get_default_transforms(dtype=torch.float32):
    return v2.Compose([v2.Resize((2560, 1920)),
                       v2.ToImage(),
                       v2.ToDtype(dtype, scale=True),
                       v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])])


def collate_curves(batch,
                   max_lines_in_page: int):
    """
    Concatenates and pads curves.
    """
    return {'image': default_collate([item['image'] for item in batch]),
            'tokens': torch.stack([F.pad(x['tokens'], pad=(0, 0, 0, max_lines_in_page-len(x['tokens'])), value=-1) for x in batch]),
            'curves': torch.stack([F.pad(x['curves'], pad=(0, 0, 0, max_lines_in_page-len(x['curves'])), value=-1) for x in batch])}


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
                 augmentation: bool = False,
                 batch_size: int = 1,
                 num_workers: int = 8):
        super().__init__()

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.line_token_id = 3

        self.save_hyperparameters()
        self.im_transforms = get_default_transforms()

    def setup(self, stage: str):
        """
        Actually builds the datasets.
        """
        self.train_set = BaselineSegmentationDataset(self.hparams.training_data,
                                                     im_transforms=self.im_transforms,
                                                     augmentation=self.hparams.augmentation,
                                                     pad_token_id=self.pad_token_id,
                                                     bos_token_id=self.bos_token_id,
                                                     eos_token_id=self.eos_token_id,
                                                     line_token_id=self.line_token_id)

        self.val_set = BaselineSegmentationDataset(self.hparams.evaluation_data,
                                                   im_transforms=self.im_transforms,
                                                   augmentation=False,
                                                   pad_token_id=self.pad_token_id,
                                                   bos_token_id=self.bos_token_id,
                                                   eos_token_id=self.eos_token_id,
                                                   line_token_id=self.line_token_id)

        max_lines_in_page = max(self.train_set.max_lines_in_page, self.val_set.max_lines_in_page)
        logger.info(f'Max number of lines in page: {max_lines_in_page} (limit: {self.train_set.max_lines_per_page})')

        self.collator = partial(collate_curves,
                                max_lines_in_page=min(max(self.train_set.max_lines_in_page,
                                                          self.val_set.max_lines_in_page),
                                                      self.train_set.max_lines_per_page))

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=self.collator)


class BaselineSegmentationDataset(Dataset):
    """
    Dataset for training a line segmentation model from baseline data.

    Args:
        files: Paths to binary dataset files.
        im_transforms: Function taking an PIL.Image and returning a tensor
                       suitable for forward passes.
        augmentation: Enables augmentation.
        max_pos_embeddings: Maximum number of lines to return from dataset.
                            Pages with more lines will be skipped.
    """
    def __init__(self,
                 files: Sequence[Union[str, 'PathLike']],
                 im_transforms=None,
                 augmentation: bool = False,
                 max_lines_per_page: int = 768,
                 pad_token_id: int = 0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 line_token_id: int = 3) -> None:
        self.files = files
        self.transforms = im_transforms
        self.aug = None
        self.arrow_table = None
        self.max_lines_in_page = 0
        self.max_lines_per_page = max_lines_per_page
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.line_token_id = line_token_id

        for file in files:
            with pa.memory_map(file, 'rb') as source:
                try:
                    ds_table = pa.ipc.open_file(source).read_all()
                except Exception:
                    logger.warning(f'{file} is not an arrow file')
                    continue
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'max_lines_in_page' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                self.max_lines_in_page = max(int.from_bytes(raw_metadata[b'max_lines_in_page'], 'little'), self.max_lines_in_page)
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
        # skip pages with more than max_pos_embeddings lines
        if len(page_data) + 2 > self.max_lines_per_page:
            rng = np.random.default_rng()
            idx = rng.integers(0, len(self))
            return self[idx]

        try:
            im = Image.open(io.BytesIO(im)).convert('RGB')
        except Exception:
            rng = np.random.default_rng()
            idx = rng.integers(0, len(self))
            return self[idx]

        im = self.transforms(im)

        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.from_numpy(o['image'].transpose(2, 0, 1))

        lines = [x['curve'] for x in page_data]
        lines.append(8 * [-1.])
        lines.insert(0, 8 * [0])
        lines = torch.tensor(lines)
        if self.aug:
            torch.rand_like(lines) * 0.001
        # one-hot encode cls here so we can embed curves and classes with a
        # single linear projection.
        line_cls = torch.full((len(lines),), self.line_token_id-1, dtype=torch.long)
        line_cls[0] = self.bos_token_id-1
        line_cls[-1] = self.eos_token_id-1
        line_cls = F.one_hot(line_cls, num_classes=3).float()
        return {'image': im,
                'tokens': line_cls,
                'curves': lines}

    def __len__(self) -> int:
        return len(self.arrow_table)
