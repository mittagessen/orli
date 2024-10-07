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
import math
import torch
import logging
import lightning.pytorch as L

from torch import nn
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.utilities.memory import (garbage_collection_cuda,
                                                is_oom_error)
from torch.optim import lr_scheduler
from torchmetrics.aggregation import MeanMetric

from transformer_seg.embedding import MBartScaledCurveEmbedding
from transformers import VisionEncoderDecoderModel, DonutSwinModel, MBartForCausalLM

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
                 curve_resolution=1000,
                 **kwargs):
        super().__init__()

        self.best_epoch = -1
        self.best_metric = 0.0
        self.best_model = None

        self.save_hyperparameters()

        logger.info('Creating segmentation model')
        encoder = DonutSwinModel.from_pretrained('mittagessen/transformer_seg_encoder')
        decoder = MBartForCausalLM.from_pretrained('mittagessen/transformer_seg_decoder')

        self.nn  = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)
        # We compute the embeddings manually as transformers flattens all but
        # the last dimension of the input_ids. We can't tie weights as the
        # EmbeddingBag is not reversible.
        # The number of embeddings is curve_resolution + sos/eos/pad + 1.
        embed_dim = curve_resolution + 4
        self.token_embeddings = MBartScaledCurveEmbedding(embed_dim,
                                                          self.nn.config.decoder.d_model,
                                                          padding_idx=self.nn.config.decoder.pad_token_id,
                                                          embed_scale=math.sqrt(self.nn.config.decoder.d_model))
        self.nn.decoder.model.decoder.embed_tokens = self.token_embeddings
        # project the hidden state to 8 * curve_resolution + 3 so we can afterwards do a view (N, S, 8, curve_resolution + 3)
        self.nn.decoder.set_output_embeddings(nn.Linear(self.nn.config.decoder.d_model, embed_dim * 8, bias=False))

        self.nn.config.decoder.vocab_size = embed_dim
        self.nn.config.decoder_start_token_id = self.nn.config.decoder.bos_token_id
        self.nn.config.eos_token_id = self.nn.config.decoder.eos_token_id
        self.nn.config.pad_token_id = self.nn.config.decoder.pad_token_id
        self.eos_token_id = self.nn.config.decoder.eos_token_id

        self.nn.train()

        self.criterion = nn.CrossEntropyLoss()

        self.val_mean = MeanMetric()

    def forward(self, x):
        return self.nn(pixel_values=x)

    def _shift_right(self, input_ids: torch.LongTensor):
        """
        Shifts a (B, S, 8)-shaped tensor to the right and preprends an SOS token.
        """
        decoder_start_token_id = self.nn.config.decoder_start_token_id
        pad_token_id = self.nn.config.decoder.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:, :] = input_ids[..., :-1, :].clone()
        shifted_input_ids[..., 0, :] = decoder_start_token_id

        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def _step(self, batch):
        try:
            shifted_curves = self._shift_right(batch['target'])
            inputs_embeds = self.token_embeddings(shifted_curves)
            output = self.nn(pixel_values=batch['image'], decoder_inputs_embeds=inputs_embeds)
            b, s, _ = output.logits.shape
            logits = output.logits.view(b, s, 8, self.nn.decoder.config.vocab_size)
            return self.criterion(logits.view(-1, self.nn.decoder.config.vocab_size), batch['target'].view(-1))
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
        self.log('val_metric', self.val_mean.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_mean.reset()
        self.log('global_step', self.global_step, on_step=False, on_epoch=True, prog_bar=False, logger=True)

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
                                                     self.nn.parameters(),
                                                     loss_tracking_mode='max')

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


def _configure_optimizer_and_lr_scheduler(hparams, params, loss_tracking_mode='max'):
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
