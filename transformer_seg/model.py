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
import torch.nn.functional as F

from torch import nn
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.memory import (garbage_collection_cuda,
                                                is_oom_error)
from torch.optim import lr_scheduler
from torchmetrics.aggregation import MeanMetric

from transformer_seg.decoder import MBartForCurveRegression
from transformers import DonutSwinModel

logger = logging.getLogger(__name__)


class SegmentationModel(L.LightningModule):
    """
    A LightningModule encapsulating the training setup for a text
    recognition model.
    """
    def __init__(self,
                 quit='fixed',
                 lag=10,
                 optimizer='AdamW',
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-3,
                 schedule='cosine',
                 step_size=10,
                 gamma=0.1,
                 rop_factor=0.1,
                 rop_patience=5,
                 cos_t_max=30,
                 cos_min_lr=1e-4,
                 warmup=15000,
                 **kwargs):
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        self.save_hyperparameters()

        logger.info('Creating segmentation model')
        self.model = nn.ModuleDict({'encoder': DonutSwinModel.from_pretrained('mittagessen/transformer_seg_encoder'),
                                    'decoder': MBartForCurveRegression.from_pretrained('mittagessen/reg_transformer_seg_decoder')})
        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
        self.model.train()

        self.val_mean = MeanMetric()

    def forward(self, x):
        return self.model(pixel_values=x)

    def _step(self, batch):
        try:
            # s is max_line_len+1 to make space for first cls, curve tuple.
            hidden_state = self.model.encoder(pixel_values=batch['image']).last_hidden_state
            # decoder shifts targets internally to right
            output = self.model.decoder(labels=batch['target'], encoder_hidden_states=hidden_state)
            # split up objectness scores and curve regressions as we only
            # compute the regression loss on the non-padded part of the lines.
            class_target = batch['target'][..., 0]
            class_pred = output.logits[..., 0]
            curve_target = batch['target'][..., 1:]
            curve_pred = output.logits[..., 1:]
            loss_cls = F.binary_cross_entropy_with_logits(class_pred.view(-1), class_target.view(-1))
            target_mask = class_target.bool()
            loss_curves = F.binary_cross_entropy_with_logits(curve_pred[target_mask].view(-1), curve_target[target_mask.bool()].view(-1))
            return loss_cls + loss_curves
        except RuntimeError as e:
            if is_oom_error(e):
                logger.warning('Out of memory error in trainer. Skipping batch and freeing caches.')
                garbage_collection_cuda()
            else:
                raise

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        if loss:
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        if loss:
            self.val_mean.update(loss)
        return loss

    def on_validation_epoch_end(self):
        if not trainer.sanity_checking:
            self.log('val_metric', self.val_mean.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.val_mean.reset()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Drops tensors with mismatching sizes.
        """
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.warning(f"Skip loading parameter: {k}, "
                                   f"required shape: {model_state_dict[k].shape}, "
                                   f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def save_checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def configure_callbacks(self):
        callbacks = []
        if self.hparams.quit == 'early':
            callbacks.append(EarlyStopping(monitor='val_accuracy',
                                           mode='max',
                                           patience=self.hparams.lag,
                                           stopping_threshold=1.0))

        return callbacks

    # configuration of optimizers and learning rate schedulers
    # --------------------------------------------------------
    #
    # All schedulers are created internally with a frequency of step to enable
    # batch-wise learning rate warmup. In lr_scheduler_step() calls to the
    # scheduler are then only performed at the end of the epoch.
    def configure_optimizers(self):
        return _configure_optimizer_and_lr_scheduler(self.hparams,
                                                     self.model.parameters(),
                                                     loss_tracking_mode='min')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)

        # linear warmup between 0 and the initial learning rate `lr` in `warmup`
        # steps.
        if self.hparams.warmup and self.trainer.global_step < self.hparams.warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.hparams.warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

    def lr_scheduler_step(self, scheduler, metric):
        if not self.hparams.warmup or self.trainer.global_step >= self.hparams.warmup:
            # step OneCycleLR each batch if not in warmup phase
            if isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            # step every other scheduler epoch-wise
            elif self.trainer.is_last_batch:
                if metric is None:
                    scheduler.step()
                else:
                    scheduler.step(metric)


def _configure_optimizer_and_lr_scheduler(hparams, params, loss_tracking_mode='min'):
    optimizer = hparams.get("optimizer")
    lr = hparams.get("lr")
    momentum = hparams.get("momentum")
    weight_decay = hparams.get("weight_decay")
    schedule = hparams.get("schedule")
    gamma = hparams.get("gamma")
    cos_t_max = hparams.get("cos_t_max")
    cos_min_lr = hparams.get("cos_min_lr")
    step_size = hparams.get("step_size")
    rop_factor = hparams.get("rop_factor")
    rop_patience = hparams.get("rop_patience")
    completed_epochs = hparams.get("completed_epochs")

    # XXX: Warmup is not configured here because it needs to be manually done in optimizer_step()
    logger.debug(f'Constructing {optimizer} optimizer (lr: {lr}, momentum: {momentum})')
    if optimizer in ['Adam', 'AdamW']:
        optim = getattr(torch.optim, optimizer)(params, lr=lr, weight_decay=weight_decay)
    else:
        optim = getattr(torch.optim, optimizer)(params,
                                                lr=lr,
                                                momentum=momentum,
                                                weight_decay=weight_decay)
    lr_sched = {}
    if schedule == 'exponential':
        lr_sched = {'scheduler': lr_scheduler.ExponentialLR(optim, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'cosine':
        lr_sched = {'scheduler': lr_scheduler.CosineAnnealingLR(optim,
                                                                cos_t_max,
                                                                cos_min_lr,
                                                                last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'step':
        lr_sched = {'scheduler': lr_scheduler.StepLR(optim, step_size, gamma, last_epoch=completed_epochs-1),
                    'interval': 'step'}
    elif schedule == 'reduceonplateau':
        lr_sched = {'scheduler': lr_scheduler.ReduceLROnPlateau(optim,
                                                                mode=loss_tracking_mode,
                                                                factor=rop_factor,
                                                                patience=rop_patience),
                    'interval': 'step'}
    elif schedule != 'constant':
        raise ValueError(f'Unsupported learning rate scheduler {schedule}.')

    ret = {'optimizer': optim}
    if lr_sched:
        ret['lr_scheduler'] = lr_sched

    if schedule == 'reduceonplateau':
        lr_sched['monitor'] = 'val_accuracy'
        lr_sched['strict'] = False
        lr_sched['reduce_on_plateau'] = True

    return ret
