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

from orli.modules.optimizer import MuonAdamW
from torchmetrics.aggregation import MeanMetric
from torch.utils.data import DataLoader
from typing import Optional, Union, TYPE_CHECKING

from kraken.models import create_model

from torchvision.ops import sigmoid_focal_loss

from orli.modules.bezier import sample_bezier_curve
from orli.dataset import (get_default_transforms, BaselineSegmentationDataset,
                          collate_curves, _validation_worker_init_fn)
from orli.configs import (OrliSegmentationTrainingConfig,
                          OrliSegmentationTrainingDataConfig,
                          OrliSegmentationTestConfig)
from orli.metrics import evaluate_page, aggregate_metrics

logger = logging.getLogger(__name__)
CURVE_STATS_INTERVAL = 200


if TYPE_CHECKING:
    from kraken.models import BaseModel
    from os import PathLike


def _orli_model_kwargs(config, image_size):
    return {'image_size': image_size,
            'anchors': config.anchors,
            'fourier_features': getattr(config, 'fourier_features', True),
            'logit_refinement': getattr(config, 'logit_refinement', True),
            'encoder_name': getattr(config, 'encoder_name', 'convnextv2_tiny'),
            'encoder_idxs': getattr(config, 'encoder_idxs', (1, 2, 3)),
            'neck_type': getattr(config, 'neck_type', 'simple'),
            'neck_num_layers': getattr(config, 'neck_num_layers', 1),
            'neck_num_heads': getattr(config, 'neck_num_heads', 8),
            'neck_hidden_dim': getattr(config, 'neck_hidden_dim', 256),
            'neck_use_encoder_idx': getattr(config, 'neck_use_encoder_idx', None),
            'neck_output_ds_factors': getattr(config, 'neck_output_ds_factors', None),
            'neck_norm': getattr(config, 'neck_norm', 'group'),
            'neck_ffn_dim': getattr(config, 'neck_ffn_dim', 1024),
            'neck_dropout': getattr(config, 'neck_dropout', 0.0),
            'neck_fusion_depth': getattr(config, 'neck_fusion_depth', 2)}


@torch.compile()
def model_step(model,
               curve_criterion,
               batch,
               teacher_force_anchors: bool = True):

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

    valid_tokens_mask = target_tokens[..., 0] != -1
    valid_curves_mask = target_curves[..., 0] != -1
    num_token_targets = valid_tokens_mask.sum().clamp_min(1)
    num_curve_targets = valid_curves_mask.sum().clamp_min(1)

    # precompute target class indices, remapping LINE tokens for multi-anchor
    target_cls_indices = target_tokens.argmax(dim=-1)  # [b, s]
    target_anchor_idx = None
    if model.nn['regressor'].num_anchors > 1:
        anchor_table = model.nn['regressor'].curve_anchors  # [num_anchors, 8]
        valid_gt = target_curves[valid_curves_mask]  # [M, 8]
        full_anchor_idx = torch.full_like(target_cls_indices, -1)
        if valid_gt.numel():
            dists = (valid_gt.unsqueeze(1) - anchor_table.unsqueeze(0)).abs().sum(-1)  # [M, N]
            nearest = dists.argmin(dim=-1)  # [M]
            full_anchor_idx[valid_curves_mask] = nearest
        line_mask = target_cls_indices == 3
        target_cls_indices[line_mask] = 3 + full_anchor_idx[line_mask].clamp(min=0)
        if teacher_force_anchors:
            target_anchor_idx = full_anchor_idx

    # our tokens already contain BOS/EOS tokens so we just run it
    # through the model after replacing ignored indices.
    tokens.masked_fill_(tokens == -1.0, 0)
    curves.masked_fill_(curves == -1.0, 0)

    logits = model(tokens=torch.cat([tokens, curves], dim=-1),
                   encoder_input=batch['image'],
                   target_anchor_idx=target_anchor_idx)
    curve_losses = None
    num_cls = logits['tokens'].shape[-1]
    num_iters = logits['curves'].shape[0]

    # valid targets (computed once)
    batch_target_cls = target_cls_indices[valid_tokens_mask]
    batch_target_curves = target_curves[valid_curves_mask]

    # one-hot targets for sigmoid focal loss
    target_cls_onehot = torch.nn.functional.one_hot(batch_target_cls.reshape(-1),
                                                     num_cls).to(dtype=logits['tokens'].dtype)

    for pred_curves in logits['curves']:
        pred_curves = pred_curves[valid_curves_mask]

        # sample points from curves
        pred_points = sample_bezier_curve(pred_curves)
        target_points = sample_bezier_curve(batch_target_curves)
        curve_points_loss = curve_criterion(pred_points, target_points) / num_curve_targets
        curve_ctrl_loss = curve_criterion(pred_curves, batch_target_curves) / num_curve_targets
        curve_loss = curve_points_loss + curve_ctrl_loss
        curve_losses = curve_loss if curve_losses is None else curve_losses + curve_loss

    # classification loss only at final iteration
    pred_tokens = logits['tokens'][-1][valid_tokens_mask]
    cls_loss = sigmoid_focal_loss(pred_tokens.reshape(-1, num_cls),
                                  target_cls_onehot,
                                  alpha=0.25,
                                  gamma=2.0,
                                  reduction='sum')
    cls_loss = cls_loss / num_token_targets

    curve_losses = curve_losses / num_iters
    losses = cls_loss + curve_losses
    return (losses, cls_loss, curve_losses)


class OrliSegmentationDataModule(L.LightningDataModule):
    def __init__(self, data_config: OrliSegmentationTrainingDataConfig):
        super().__init__()

        self.bos_token_id = 1
        self.eos_token_id = 2
        self.line_token_id = 3

        self.save_hyperparameters()
        self.hparams.data_config.val_batch_size = data_config.batch_size if not data_config.val_batch_size else data_config.val_batch_size

        im_transforms = get_default_transforms(image_size=data_config.image_size,
                                               normalize=False)

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
        loader_kwargs = {}
        if self.hparams.data_config.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 4
        return DataLoader(self.train_set,
                          batch_size=self.hparams.data_config.batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=self.collator,
                          **loader_kwargs)

    def val_dataloader(self):
        loader_kwargs = {}
        if self.hparams.data_config.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 4
        return DataLoader(self.val_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.val_batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=self.collator,
                          **loader_kwargs)

    def test_dataloader(self):
        loader_kwargs = {}
        if self.hparams.data_config.num_workers > 0:
            loader_kwargs['prefetch_factor'] = 4
        return DataLoader(self.test_set,
                          shuffle=False,
                          batch_size=self.hparams.data_config.val_batch_size,
                          num_workers=self.hparams.data_config.num_workers,
                          pin_memory=True,
                          worker_init_fn=_validation_worker_init_fn,
                          collate_fn=self.collator,
                          **loader_kwargs)


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

            if 'segmentation' not in self.net.model_type:
                raise ValueError(f'Model {model} is of type {self.net.model_type} while `segmentation` is expected.')
        else:
            self.net = None

        self.val_cls_mean = MeanMetric()
        self.val_curve_mean = MeanMetric()
        self.curve_criterion = torch.nn.L1Loss(reduction='sum')
        self.test_tolerance = 10.0
        self.test_match_threshold = 0.5
        self.test_results = {}
        self._test_page_metrics = []
        self._test_config = None

    def forward(self, x):
        return self.model(pixel_values=x)

    def training_step(self, batch, batch_idx):
        loss, cls_loss, curve_loss = model_step(self.net,
                                                self.curve_criterion,
                                                batch,
                                                teacher_force_anchors=True)
        self.log('train_loss',
                 loss,
                 batch_size=batch['tokens'].shape[0],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        self.log('train_cls_loss',
                 cls_loss,
                 batch_size=batch['tokens'].shape[0],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        self.log('train_curve_loss',
                 curve_loss,
                 batch_size=batch['tokens'].shape[0],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        num_token_targets = (batch['tokens'][..., 1:, 0] != -1).sum().clamp_min(1)
        num_curve_targets = (batch['curves'][..., 1:, 0] != -1).sum().clamp_min(1)

        loss, cls_loss, curve_loss = model_step(self.net,
                                                self.curve_criterion,
                                                batch,
                                                teacher_force_anchors=getattr(self.hparams.config,
                                                                              'teacher_force_anchors',
                                                                              True))
        self.val_cls_mean.update(cls_loss.detach(),
                                 weight=num_token_targets.to(device=cls_loss.device, dtype=cls_loss.dtype))
        self.val_curve_mean.update(curve_loss.detach(),
                                   weight=num_curve_targets.to(device=curve_loss.device, dtype=curve_loss.dtype))
        return loss

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            val_cls_loss = self.val_cls_mean.compute()
            val_curve_loss = self.val_curve_mean.compute()
            self.log('val_cls_loss',
                     val_cls_loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True)
            self.log('val_curve_loss',
                     val_curve_loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True)
            self.log('val_metric',
                     val_cls_loss + val_curve_loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True,
                     sync_dist=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_cls_mean.reset()
        self.val_curve_mean.reset()

    def configure_test(self, test_config: OrliSegmentationTestConfig):
        self._test_config = test_config
        self.test_tolerance = test_config.tolerance
        self.test_match_threshold = test_config.match_threshold

    def on_test_start(self):
        if self._test_config is None:
            raise RuntimeError('Test configuration missing. Call configure_test() before running tests.')
        if self.net is None:
            raise RuntimeError('Model is not initialized.')
        self._test_page_metrics = []
        self.test_results = {}
        self.net.prepare_for_inference(self._test_config)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images = batch['image']
        pred_curves = self.net.predict_curves(images)
        if pred_curves.dim() == 2:
            pred_curves = pred_curves.unsqueeze(0)

        image_size = (images.shape[-1], images.shape[-2])

        for idx in range(images.shape[0]):
            pred = pred_curves[idx]
            pred = pred[pred.abs().sum(dim=-1) > 0]
            pred = pred.cpu().float()

            gt = batch['curves'][idx]
            if gt.shape[0] > 0:
                gt = gt[1:]
            gt = gt[gt[..., 0] != -1]
            gt = gt.cpu().float()

            metrics = evaluate_page(pred_curves=pred,
                                    gt_curves=gt,
                                    image_size=image_size,
                                    tol=self.test_tolerance,
                                    match_threshold=self.test_match_threshold)
            self._test_page_metrics.append(metrics)

    def on_test_epoch_end(self):
        if not self._test_page_metrics:
            self.test_results = {}
            return
        self.test_results = aggregate_metrics(self._test_page_metrics)

    def setup(self, stage: Optional[str] = None):
        logger.info('Creating segmentation model')

        if stage in [None, 'fit']:
            if self.net is None:
                self.net = create_model('OrliModel',
                                        **_orli_model_kwargs(self.hparams.config,
                                                             self.trainer.datamodule.hparams.data_config.image_size))

            if self.hparams.config.freeze_encoder:
                for param in self.net.nn['encoder'].parameters():
                    param.requires_grad = False
                for param in self.net.nn['adapter'].parameters():
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
        cfg = checkpoint['_module_config']
        self.net = create_model('OrliModel',
                                **_orli_model_kwargs(cfg, data_config.image_size))

    @classmethod
    def load_from_weights(cls,
                          path: Union[str, 'PathLike'],
                          config: OrliSegmentationTrainingConfig) -> 'OrliSegmentationModel':
        """
        Initializes the module from a model weights file.
        """
        from kraken.models import load_models
        from orli.orli import OrliModel
        models = load_models(path, tasks=['segmentation'])
        model = None
        for candidate in models:
            if isinstance(candidate, OrliModel):
                model = candidate
                break
        if model is None:
            raise ValueError('No OrliModel found in weights file.')
        return cls(config=config, model=model)

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
        # Selective learning rates: lower LR for pretrained encoder, full LR otherwise.
        # Muon only works with 2D parameters (weight matrices). Everything else
        # (biases, norms, embeddings, 1D params) must use AdamW.
        lr_map = {}

        # Encoder (pretrained) - lower LR when unfrozen
        encoder_lr = self.hparams.config.lrate * 0.1
        for p in self.net.nn['encoder'].parameters():
            if p.requires_grad:
                lr_map[id(p)] = encoder_lr

        # Regressor - higher LR
        regressor_lr = self.hparams.config.lrate * 5.0
        for p in self.net.nn['regressor'].parameters():
            if p.requires_grad:
                lr_map[id(p)] = regressor_lr

        # Everything else (decoder + neck) - full LR
        for p in self.net.parameters():
            if p.requires_grad and id(p) not in lr_map:
                lr_map[id(p)] = self.hparams.config.lrate

        # Split into Muon (2D) and AdamW (non-2D) param groups, keyed by lr
        muon_by_lr = {}
        adam_by_lr = {}
        for p in self.net.parameters():
            if not p.requires_grad:
                continue
            target_lr = lr_map[id(p)]
            if p.ndim == 2:
                muon_by_lr.setdefault(target_lr, []).append(p)
            else:
                adam_by_lr.setdefault(target_lr, []).append(p)

        muon_groups = [{'params': params, 'lr': lr, 'initial_lr': lr}
                       for lr, params in muon_by_lr.items()]
        adam_groups = [{'params': params, 'lr': lr, 'initial_lr': lr}
                      for lr, params in adam_by_lr.items()]

        optimizer = MuonAdamW(
            muon_params=muon_groups,
            adam_params=adam_groups,
            muon_kwargs={'weight_decay': self.hparams.config.weight_decay},
            adam_kwargs={'weight_decay': self.hparams.config.weight_decay,
                        'betas': (0.9, 0.999)},
        )

        # Configure learning rate scheduler
        len_train_set = len(self.trainer.datamodule.train_set)
        batch_size = self.trainer.datamodule.hparams.data_config.batch_size
        accumulate = self.hparams.config.accumulate_grad_batches
        num_devices = max(1, self.trainer.num_devices)
        steps_per_epoch = len_train_set // (batch_size * accumulate * num_devices)

        if self.hparams.config.schedule == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.config.cos_t_max * steps_per_epoch,
                eta_min=self.hparams.config.cos_min_lr
            )
        elif self.hparams.config.schedule == 'reduceonplateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5
            )
        elif self.hparams.config.schedule == 'exponential':
            scheduler = lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.hparams.config.exp_gamma
            )
        elif self.hparams.config.schedule == 'step':
            scheduler = lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.config.step_size,
                gamma=self.hparams.config.step_gamma
            )
        elif self.hparams.config.schedule == '1cycle':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.config.lrate,
                total_steps=self.hparams.config.epochs * steps_per_epoch
            )
        else:
            scheduler = lr_scheduler.ConstantLR(optimizer, factor=1.0)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_metric',
                'interval': 'step',
                'frequency': 1,
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate in `warmup` steps.
        # Each param group has its own target lr (encoder has lower lr).
        if self.hparams.config.warmup and self.trainer.global_step < self.hparams.config.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.config.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * pg.get("initial_lr", self.hparams.config.lrate)

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.config.warmup or self.trainer.global_step >= self.hparams.config.warmup:
            if isinstance(scheduler, (lr_scheduler.OneCycleLR, lr_scheduler.CosineAnnealingLR)):
                scheduler.step()
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)
