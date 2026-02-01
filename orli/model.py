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
Training loop interception helpers
"""
import torch
import logging
import lightning.pytorch as L

from functools import partial
from lightning.pytorch.callbacks import EarlyStopping
from torch.optim import lr_scheduler
from torchmetrics.aggregation import MeanMetric
from torch.utils.data import DataLoader
from typing import Optional, Union, TYPE_CHECKING

from kraken.models import create_model
from kraken.train.utils import configure_optimizer_and_lr_scheduler

from orli.modules.bezier import sample_bezier_curve
from orli.dataset import (get_default_transforms, BaselineSegmentationDataset,
                          collate_curves, _validation_worker_init_fn)
from orli.configs import OrliSegmentationTrainingConfig, OrliSegmentationTrainingDataConfig

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from kraken.models import BaseModel
    from os import PathLike


@torch.compile()
def model_step(model,
               cls_criterion,
               curve_criterion,
               batch):

    tokens = batch['tokens']
    curves = batch['curves']
    # shift the tokens to create targets
    ignore_idxs_tokens = torch.full((tokens.shape[0], 1, tokens.shape[2]),
                                    -1,
                                    dtype=tokens.dtype, device=tokens.device)
    ignore_idxs_curves = torch.full((curves.shape[0], 1, curves.shape[2]),
                                    -1,
                                    dtype=curves.dtype, device=curves.device)

    target_tokens = torch.hstack((tokens[..., 1:, :], ignore_idxs_tokens))
    target_curves = torch.hstack((curves[..., 1:, :], ignore_idxs_curves))
    # our tokens already contain BOS/EOS tokens so we just run it
    # through the model after replacing ignored indices.
    tokens.masked_fill_(tokens == -1.0, 0)
    curves.masked_fill_(curves == -1.0, 0)

    logits = model(tokens=torch.cat([tokens, curves], dim=-1), encoder_input=batch['image'])
    losses = None
    num_lines = target_tokens[target_tokens != -1].view(-1, 4).shape[0]

    # match ground truth curves to anchors
    valid_targets_mask = target_curves[..., 0] != -1
    valid_target_curves = target_curves[valid_targets_mask]
    anchors = model.nn['regressor'].curve_anchors
    # l1 distance between each target curve and each anchor
    l1_dist = torch.cdist(valid_target_curves, anchors, p=1)
    # find best anchor for each target curve
    best_anchor_idx = torch.argmin(l1_dist, dim=-1)

    for pred_curves, pred_tokens in zip(logits['curves'], logits['tokens']):
        # filter out ignored indices from predictions
        pred_curves = pred_curves[valid_targets_mask]
        pred_tokens = pred_tokens[valid_targets_mask]

        # valid targets
        batch_target_curves = target_curves[valid_targets_mask]
        batch_target_tokens = target_tokens[valid_targets_mask]

        # create target for classification loss
        item_indices = torch.arange(pred_tokens.shape[0])
        target_cls_idx = torch.zeros(pred_tokens.shape[0], pred_tokens.shape[1],
                                     dtype=torch.long, device=pred_tokens.device)
        target_cls_idx[item_indices, best_anchor_idx] = batch_target_tokens.argmax(dim=-1)
        cls_loss = cls_criterion(pred_tokens.reshape(-1, pred_tokens.shape[-1]),
                                 target_cls_idx.reshape(-1))

        # select predictions for best anchors for curve loss
        selected_pred_curves = pred_curves[torch.arange(pred_curves.shape[0]), best_anchor_idx]

        # sample points from curves
        pred_points = sample_bezier_curve(selected_pred_curves)
        target_points = sample_bezier_curve(batch_target_curves)
        curve_loss = curve_criterion(pred_points, target_points)

        _loss = 2 * cls_loss + 5 * curve_loss
        losses = _loss if not losses else losses + _loss
    return losses / (logits['curves'].shape[0] * num_lines)


class OrliSegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_config: OrliSegmentationTrainingDataConfig):
        super().__init__()

        self.bos_token_id = 1
        self.eos_token_id = 2
        self.line_token_id = 3

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size

        im_transforms = get_default_transforms(image_size=data_config.image_size)

        if data_config.training_data and data_config.evaluation_data:
            self.train_set = BaselineSegmentationDataset(data_config.training_data,
                                                         im_transforms=im_transforms,
                                                         augmentation=data_config.augment,
                                                         bos_token_id=self.bos_token_id,
                                                         eos_token_id=self.eos_token_id,
                                                         line_token_id=self.line_token_id)
            self.val_set = BaselineSegmentationDataset(data_config.evaluation_data,
                                                       im_transforms=im_transforms,
                                                       bos_token_id=self.bos_token_id,
                                                       eos_token_id=self.eos_token_id,
                                                       line_token_id=self.line_token_id)

            if len(self.train_set) == 0:
                raise ValueError('No valid training data provided. Please add some.')
            if len(self.val_set) == 0:
                raise ValueError('No valid validation data provided. Please add some.')
            max_lines_in_page = max(self.train_set.max_lines_in_page, self.val_set.max_lines_in_page)
            line_limit = self.train_set.max_lines_per_page
        elif data_config.test_data:
            self.test_set = BaselineSegmentationDataset(data_config.test_data,
                                                        im_transforms=im_transforms,
                                                        bos_token_id=self.bos_token_id,
                                                        eos_token_id=self.eos_token_id,
                                                        line_token_id=self.line_token_id)

            max_lines_in_page = self.test_set.max_lines_in_page
            line_limit = self.test_set.max_lines_per_page

            if len(self.test_set) == 0:
                raise ValueError('No valid test data provided. Please add some.')
        else:
            raise ValueError('Invalid specification of training/evaluation/test data.')

        logger.info(f'Max number of lines in page: {max_lines_in_page} (limit: {line_limit})')

        self.collator = partial(collate_curves,
                                max_lines_in_page=min(max_lines_in_page, line_limit))

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.hparams.data_config.line_class_mapping = dict(self.hparams.data_config.line_class_mapping)
            self.hparams.data_config.region_class_mapping = dict(self.hparams.data_config.region_class_mapping)

    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=self.collator)

    def test_dataloader(self):
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=self.collator)


class OrliSegmentationModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a text
    recognition model.
    """
    def __init__(self,
                 config: OrliSegmentationTrainingConfig,
                 model: Optional['BaseModel'] = None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        if model:
            self.net = model

            if self.net.model_type not in [None, 'segmentation']:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `segmentation` is expected.')
        else:
            self.net = None

        self.val_mean = MeanMetric()
        self.curve_criterion = torch.nn.L1Loss(reduction='sum')
        self.cls_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        return self.model(pixel_values=x)

    def training_step(self, batch, batch_idx):
        loss = model_step(self.net,
                          self.cls_criterion,
                          self.curve_criterion,
                          batch)
        self.log('train_loss',
                 loss,
                 batch_size=batch['tokens'].shape[0],
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = model_step(self.net,
                          self.cls_criterion,
                          self.curve_criterion,
                          batch)
        self.val_mean.update(loss)
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log('val_metric', self.val_mean.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_mean.reset()

    def setup(self, stage: Optional[str] = None):
        logger.info('Creating segmentation model')

        if stage in [None, 'fit']:
            if self.net is None:
                self.net = create_model('OrliModel',
                                        image_size=self.trainer.datamodule.hparams.data_config.image_size,
                                        anchors_path=self.hparams.config.anchors_path)

            if self.hparams.config.freeze_encoder:
                for param in self.net.encoder.parameters():
                    param.requires_grad = False
                for param in self.net.adapter.parameters():
                    param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        """
        Save hyperparameters a second time so we can set parameters that
        shouldn't be overwritten in on_load_checkpoint.
        """
        checkpoint['_module_config'] = self.hparams.config

    def on_load_checkpoint(self, checkpoint):
        """
        Reconstruct the model from the spec here and not in setup() as
        otherwise the weight loading will fail.
        """
        if not isinstance(checkpoint['_module_config'], OrliSegmentationTrainingConfig):
            raise ValueError('Checkpoint is not an orli model.')

        data_config = checkpoint['datamodule_hyper_parameters']['data_config']
        self.net = create_model('OrliModel',
                                image_size=data_config.image_size,
                                anchors_path=checkpoint['_module_config'].anchors_path)

    @classmethod
    def load_from_repo(cls,
                       id: str,
                       config: OrliSegmentationTrainingConfig):
        """
        Loads weights from HTRMoPo.
        """
        from htrmopo import get_model

        model_path = get_model(id) / 'model.safetensors'
        return cls.load_from_weights(path=model_path, config=config)

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: OrliSegmentationTrainingConfig) -> 'OrliSegmentationModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        models = load_models(path, tasks=['segmentation'])
        if len(models) != 1:
            raise ValueError(f'Found {len(models)} segmentation models in model file.')
        return cls(config=config, model=models[0])

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.config.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_metric',
                                           mode='min',
                                           patience=self.hparams.config.lag,
                                           stopping_threshold=0.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return configure_optimizer_and_lr_scheduler(self.hparams.config,
                                                    self.net.parameters(),
                                                    len_train_set=len(self.trainer.datamodule.train_set),
                                                    loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lrate` in `warmup`
        # steps.
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.config.lrate

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
