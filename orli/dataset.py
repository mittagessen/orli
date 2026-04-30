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
import gc
import torch
import ctypes
import tempfile
import torch.nn.functional as F

import numpy as np
import pyarrow as pa

from torchvision.transforms import v2
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Union, Sequence

from torch.utils.data import Dataset

from PIL import Image

from torch.utils.data import default_collate

from orli.modules.baseline import (DEFAULT_NUM_BASELINE_POINTS,
                                   baseline_polyline_dim,
                                   curve_vector_dim,
                                   fixed_arc_length_resample,
                                   polyline_to_curve_vector)

if TYPE_CHECKING:
    from os import PathLike

__all__ = ['BaselineSegmentationDataset', 'compile']

import logging

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 20000 ** 2

TORCHVISION_PAGE_IMAGE_FORMATS = frozenset({'JPEG', 'PNG', 'WEBP', 'GIF'})
TRANSCODE_PAGE_IMAGE_FORMATS = (('JPEG', {'quality': 95}),
                                ('PNG', {}))


def _encode_page_image(path: Union[str, 'PathLike'],
                       resize: Optional[tuple[int, int]] = None) -> bytes:
    with Image.open(path) as image:
        image_format = image.format
        if resize is None and image_format in TORCHVISION_PAGE_IMAGE_FORMATS:
            with open(path, 'rb') as fp:
                return fp.read()

        image = image.convert('RGB')
        if resize:
            image = image.resize((resize[1], resize[0]), Image.LANCZOS)

        encoded_variants = []
        for fmt, save_kwargs in TRANSCODE_PAGE_IMAGE_FORMATS:
            image_buffer = io.BytesIO()
            image.save(image_buffer, format=fmt, **save_kwargs)
            encoded_variants.append(image_buffer.getvalue())

    return min(encoded_variants, key=len)


def _to_polyline(baseline,
                 im_size: tuple[int, int],
                 clamp_tolerance: float = 0.02) -> Optional[pa.Scalar]:
    points = np.asarray(baseline, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        logger.info('Skipping line with fewer than two baseline points')
        return None

    points = points / np.asarray(im_size, dtype=np.float32)
    if not np.all(np.isfinite(points)):
        logger.info('Skipping line with non-finite baseline points')
        return None
    if np.any(points < -clamp_tolerance) or np.any(points > 1 + clamp_tolerance):
        logger.info('Skipping line with baseline points substantially outside image bounds')
        return None

    points = np.clip(points, 0.0, 1.0)
    if np.linalg.norm(points[-1] - points[0]) < 1e-6:
        logger.info('Skipping line with degenerate baseline')
        return None

    point_type = pa.list_(pa.float32(), 2)
    return pa.scalar(points.tolist(), type=pa.list_(point_type))


def compile(files: Optional[list[Union[str, 'PathLike']]] = None,
            output_file: Union[str, 'PathLike'] = None,
            resize: Optional[tuple[int, int]] = None,
            allow_textless: bool = False,
            callback: Callable[[int, int], None] = lambda chunk, lines: None) -> dict[str, int]:
    """
    Compiles XML facsimile files into an Arrow dataset with normalized baseline
    polylines in implicit reading order.
    """
    from kraken.lib.xml import XMLPage

    if files is None:
        files = []

    point_type = pa.list_(pa.float32(), 2)
    line_struct = pa.struct([('polyline', pa.list_(point_type))])
    page_struct = pa.struct([('im', pa.binary()),
                             ('lines', pa.list_(line_struct))])

    num_lines = 0
    num_pages = 0
    max_lines_in_page = 0
    schema = pa.schema([('pages', page_struct)])

    callback(0, len(files))

    with tempfile.NamedTemporaryFile() as tmpfile:
        with pa.OSFile(tmpfile.name, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for file in files:
                    try:
                        page = XMLPage(file)
                        seg = page.to_container()
                        im_bytes = _encode_page_image(seg.imagename, resize=resize)
                    except Exception:
                        logger.exception(f'Skipping unreadable XML page {file}')
                        callback(1, len(files))
                        continue

                    page_data = []
                    for line in seg.lines:
                        try:
                            text = line.text or ''
                            if not text and not allow_textless:
                                logger.info('Skipping textless line')
                                continue
                            if not line.baseline:
                                logger.info('No baseline given for line')
                                continue
                            polyline = _to_polyline(line.baseline, page.image_size)
                            if polyline is None:
                                continue
                            page_data.append(pa.scalar({'polyline': polyline}, line_struct))
                            num_lines += 1
                        except Exception:
                            logger.exception('Skipping invalid line')
                            continue

                    if page_data:
                        ar = pa.array([pa.scalar({'im': im_bytes,
                                                  'lines': page_data}, page_struct)], page_struct)
                        writer.write(pa.RecordBatch.from_arrays([ar], schema=schema))
                        max_lines_in_page = max(len(page_data), max_lines_in_page)
                        num_pages += 1
                    callback(1, len(files))

        with pa.memory_map(tmpfile.name, 'rb') as source:
            metadata = {'num_lines': num_lines.to_bytes(4, 'little'),
                        'num_pages': num_pages.to_bytes(4, 'little'),
                        'max_lines_in_page': max_lines_in_page.to_bytes(4, 'little')}
            schema = schema.with_metadata(metadata)
            ds_table = pa.ipc.open_file(source).read_all()
            new_table = ds_table.replace_schema_metadata(metadata)
            with pa.OSFile(output_file, 'wb') as sink:
                with pa.ipc.new_file(sink, schema=schema) as writer:
                    for batch in new_table.to_batches():
                        writer.write(batch)

    return {'num_lines': num_lines,
            'num_pages': num_pages,
            'max_lines_in_page': max_lines_in_page}


def get_default_transforms(image_size, dtype=torch.float32, normalize: bool = True):
    transforms = [v2.Resize(image_size),
                  v2.ToImage(),
                  v2.ToDtype(dtype, scale=True)]
    if normalize:
        transforms.append(v2.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225)))
    return v2.Compose(transforms)


def collate_curves(batch,
                   max_lines_in_page: int):
    """
    Concatenates and pads baseline vectors and polyline targets.
    """
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    gc.collect()
    return {'image': default_collate([item['image'] for item in batch]),
            'tokens': torch.stack([F.pad(x['tokens'], pad=(0, 0, 0, max_lines_in_page-len(x['tokens'])), value=-1) for x in batch]),
            'curves': torch.stack([F.pad(x['curves'], pad=(0, 0, 0, max_lines_in_page-len(x['curves'])), value=-1) for x in batch]),
            'polylines': torch.stack([F.pad(x['polylines'], pad=(0, 0, 0, max_lines_in_page-len(x['polylines'])), value=-1) for x in batch])}


def _validation_worker_init_fn(worker_id):
    """ Fix random seeds so that augmentation always produces the same
        results when validating. Temporarily increase the logging level
        for lightning because otherwise it will display a message
        at info level about the seed being changed. """
    from lightning.pytorch import seed_everything
    seed_everything(42)


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
                 normalize_image: bool = True,
                 augmentation: bool = False,
                 max_lines_per_page: int = 768,
                 baseline_num_points: int = DEFAULT_NUM_BASELINE_POINTS,
                 direct_point_regression: bool = False,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 line_token_id: int = 3) -> None:
        self.files = files
        self.transforms = im_transforms
        self.aug = None
        self.arrow_table = None
        self.max_lines_in_page = 0
        self.max_lines_per_page = max_lines_per_page
        self.baseline_num_points = int(baseline_num_points)
        self.direct_point_regression = bool(direct_point_regression)
        self.baseline_dim = curve_vector_dim(self.baseline_num_points,
                                             self.direct_point_regression)
        self.polyline_dim = baseline_polyline_dim(self.baseline_num_points)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.line_token_id = line_token_id
        self.normalizer = v2.Normalize(mean=(0.485, 0.456, 0.406),
                                       std=(0.229, 0.224, 0.225)) if normalize_image else None
        self.rng = np.random.default_rng()

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
                self.max_lines_in_page = max(int.from_bytes(raw_metadata[b'max_lines_in_page'], 'little') + 2, self.max_lines_in_page)
                if not self.arrow_table:
                    self.arrow_table = ds_table
                else:
                    self.arrow_table = pa.concat_tables([self.arrow_table, ds_table])

        if augmentation:
            from orli.augmentation import DefaultAugmenter
            self.aug = DefaultAugmenter()

    def __getitem__(self, index: int):
        # just sample from a random page
        item = self.arrow_table.column('pages')[index].as_py()
        logger.debug(f'Attempting to load {item["im"]}')
        im, page_data = item['im'], item['lines']
        # skip pages with more than max_pos_embeddings lines
        if len(page_data) + 2 >= self.max_lines_per_page:
            idx = int(self.rng.integers(0, len(self)))
            return self[idx]

        try:
            im = Image.open(io.BytesIO(im)).convert('RGB')
        except Exception:
            idx = int(self.rng.integers(0, len(self)))
            return self[idx]

        im = self.transforms(im)

        polylines = []
        lines = []
        for line in page_data:
            if 'polyline' not in line:
                raise ValueError('Baseline dataset lines must contain a polyline field.')
            source_points = torch.tensor(line['polyline'], dtype=torch.float32)
            if source_points.ndim < 2 or source_points.shape[-1] != 2:
                raise ValueError(f'Expected polyline with shape [n, 2], got {tuple(source_points.shape)}.')
            points = fixed_arc_length_resample(source_points.reshape(-1, 2).clamp(0.0, 1.0),
                                               num_points=self.baseline_num_points)
            params = polyline_to_curve_vector(points,
                                              direct_point_regression=self.direct_point_regression)
            polylines.append(points.reshape(self.polyline_dim))
            lines.append(params)

        lines.append(torch.full((self.baseline_dim,), -1.0, dtype=torch.float32))
        lines.insert(0, torch.zeros(self.baseline_dim, dtype=torch.float32))
        lines = torch.stack(lines, dim=0)
        polylines.append(torch.full((self.polyline_dim,), -1.0, dtype=torch.float32))
        polylines.insert(0, torch.zeros(self.polyline_dim, dtype=torch.float32))
        polylines = torch.stack(polylines, dim=0)
        # one-hot encode cls here so we can embed curves and classes with a
        # single linear projection.
        line_cls = torch.full((len(lines),), self.line_token_id, dtype=torch.long)
        line_cls[0] = self.bos_token_id
        line_cls[-1] = self.eos_token_id
        line_cls = F.one_hot(line_cls, num_classes=4).float()

        if self.aug:
            im = im.permute((1, 2, 0)).numpy()
            o = self.aug(image=im)
            im = torch.from_numpy(o['image'].transpose(2, 0, 1))
        if self.normalizer:
            im = self.normalizer(im)

        return {'image': im,
                'tokens': line_cls,
                'curves': lines,
                'polylines': polylines}

    def __len__(self) -> int:
        return len(self.arrow_table)
