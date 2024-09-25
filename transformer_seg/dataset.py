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
import numpy as np
import lightning.pytorch as L

import tempfile
import pyarrow as pa

from typing import (TYPE_CHECKING, Callable, List, Literal, Optional, Union,
                    Sequence)

from functools import partial
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from scipy.special import comb
from shapely.geometry import LineString

from kraken.lib import functional_im_transforms as F_t
from kraken.lib.xml import XMLPage

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


def _to_curve(baseline, im_size, min_points: int = 8) -> torch.Tensor:
    """
    Converts poly(base)lines to Bezier curves and normalizes them.
    """
    baseline = np.array(baseline)
    if len(baseline) < min_points:
        ls = LineString(baseline)
        baseline = np.stack([np.array(ls.interpolate(x, normalized=True).coords)[0] for x in np.linspace(0, 1, 8)])
    # control points normalized to patch extents
    curve = np.concatenate(([baseline[0]], bezier_fit(baseline), [baseline[-1]]))/im_size
    curve = curve.flatten()
    return pa.scalar(curve, type=pa.list_(pa.float32()))


def compile(files: Optional[List[Union[str, 'PathLike']]] = None,
            output_file: Union[str, 'PathLike'] = None,
            max_side_length: int = 4000,
            reorder: Union[bool, Literal['L', 'R']] = True,
            normalize_whitespace: bool = True,
            normalization: Optional[Literal['NFD', 'NFC', 'NFKD', 'NFKC']] = None,
            callback: Callable[[int, int], None] = lambda chunk, lines: None) -> None:
    """
    Compiles a collection of XML facsimile files into a binary arrow dataset.

    Args:
        files: List of XML files
        output_file: destination to write arrow file to
        max_side_length: Max length of longest image side.
        reorder: text reordering
        normalize_whitespace: whether to normalize all whitespace to ' '
        normalization: Unicode normalization to apply to data.
        callback: progress callback
    """
    text_transforms: List[Callable[[str], str]] = []

    # pyarrow structs
    line_struct = pa.struct([('text', pa.list_(pa.int32())), ('curve', pa.list_(pa.float32()))])
    page_struct = pa.struct([('im', pa.binary()), ('lines', pa.list_(line_struct))])

    codec = ByT5Codec()

    if normalization:
        text_transforms.append(partial(F_t.text_normalize, normalization=normalization))
    if normalize_whitespace:
        text_transforms.append(F_t.text_whitespace_normalize)
        if reorder:
            if reorder in ('L', 'R'):
                text_transforms.append(partial(F_t.text_reorder, base_dir=reorder))
            else:
                text_transforms.append(F_t.text_reorder)

    num_lines = 0

    schema = pa.schema([('pages', page_struct)])

    callback(0, len(files))

    with tempfile.NamedTemporaryFile() as tmpfile:
        with pa.OSFile(tmpfile.name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for file in files:
                    try:
                        page = XMLPage(file).to_container()
                        im = Image.open(page.imagename)
                        im_size = im.size
                    except Exception:
                        continue
                    page_data = []
                    for line in page.lines:
                        try:
                            text = line.text
                            for func in text_transforms:
                                text = func(text)
                            if not text:
                                logger.info(f'Text line "{line.text}" is empty after transformations')
                                continue
                            if not line.baseline:
                                logger.info('No baseline given for line')
                                continue
                            page_data.append(pa.scalar({'text': pa.scalar(codec.encode(text).numpy()),
                                                        'curve': _to_curve(line.baseline, im_size)},
                                                       line_struct))
                            num_lines += 1
                        except Exception:
                            continue
                    if len(page_data) > 1:
                        # scale image only now
                        im = optional_resize(im, max_side_length).convert('RGB')
                        fp = io.BytesIO()
                        im.save(fp, format='png')
                        ar = pa.array([pa.scalar({'im': fp.getvalue(), 'lines': page_data}, page_struct)], page_struct)
                        writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))
                    callback(1, len(files))
        with pa.memory_map(tmpfile.name, 'rb') as source:
            metadata = {'num_lines': num_lines.to_bytes(4, 'little')}
            schema = schema.with_metadata(metadata)
            ds_table = pa.ipc.open_file(source).read_all()
            new_table = ds_table.replace_schema_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema=schema) as writer:
                    for batch in new_table.to_batches():
                        writer.write(batch)


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


def optional_resize(img: 'Image.Image', max_size: int):
    """
    Resizing that return images with the longest side below `max_size`
    unchanged.

    Args:
        img: image to resize
        max_size: maximum length of any side of the image
    """
    w, h = img.size
    img_max = max(w, h)
    if img_max > max_size:
        if w > h:
            h = int(h * max_size/w)
            w = max_size
        else:
            w = int(w * max_size/h)
            h = max_size
        return img.resize((w, h))
    else:
        return img


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
                ds_table = pa.ipc.open_file(source).read_all()
                raw_metadata = ds_table.schema.metadata
                if not raw_metadata or b'num_lines' not in raw_metadata:
                    raise ValueError(f'{file} does not contain a valid metadata record.')
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])

        if augmentation:
            from party.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __getitem__(self, index: int):
        # just sample from a random page
        item = self.arrow_table.column('pages')[index].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        im = Image.open(io.BytesIO(im))
        im = self.transforms(im)['pixel_values'][0]

        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.tensor(o['image'].transpose(2, 0, 1))

        lines = (torch.tensor([x['curve'] for x in page_data]) * self.curve_resolution).to(torch.long)
        # offset behind sos/eos/pad token indices
        lines += 3
        # append eos token
        lines = torch.cat([lines, torch.full((1, 8), self.eos_token_id)])
        return {'image': im, 'target': lines}

    def __len__(self) -> int:
        return len(self.arrow_table)


# magic lsq cubic bezier fit function from the internet.
def Mtk(n, t, k):
    return t**k * (1-t)**(n-k) * comb(n, k)


def BezierCoeff(ts):
    return [[Mtk(3, t, k) for k in range(4)] for t in ts]


def bezier_fit(bl):
    x = bl[:, 0]
    y = bl[:, 1]
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2)**0.5
    t = dt/dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(bl)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :]
    return medi_ctp
