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
import timm
import torch
import logging
import torch.nn.functional as F

from torch import nn
from lightning.fabric import Fabric
from typing import Optional, Any, Union
from collections.abc import Generator

from orli.fusion import baseline_decoder, OrliAdapter, CurveRegressionHead

from kraken.models import BaseModel

logger = logging.getLogger(__name__)


__all__ = ['OrliModel']


class OrliModel(nn.Module, BaseModel):
    """
    The transformer segmentation fusion model.

    Args:
    """
    bos_id: int = 1
    eos_id: int = 2
    user_metadata = {}
    model_type = 'segmentation'
    _kraken_min_version = '6.0.0'

    def __init__(self, **kwargs):
        super().__init__()

        adapter_num_layers = 1
        adapter_num_heads = 8
        encoder_idxs = [1, 2, 3]

        if (image_size := kwargs.get('image_size', None)) is None:
            raise ValueError('image_size argument is missing in args.')

        anchors_path = kwargs.pop('anchors_path', None)
        if not anchors_path:
            raise ValueError('anchors_path argument is missing in args.')

        self.user_metadata: dict[str, Any] = {'accuracy': [],
                                              'metrics': []}
        self.user_metadata.update(kwargs)

        logger.info('Creating segmentation model')

        encoder_model = timm.create_model('convnextv2_tiny',
                                          pretrained=True,
                                          features_only=True,
                                          out_indices=encoder_idxs)

        encoder_sizes = [(int(image_size[0]/encoder_model.feature_info.reduction(idx)),
                          int(image_size[1]/encoder_model.feature_info.reduction(idx)),
                          encoder_model.feature_info.channels(idx)) for idx in encoder_idxs]

        adapter = OrliAdapter(adapter_num_layers,
                              adapter_num_heads,
                              encoder_embed_dims=[x[2] for x in encoder_sizes],
                              encoder_sizes=[x[:2] for x in encoder_sizes],
                              decoder_embed_dim=576)

        decoder_model = baseline_decoder(encoder_sizes=adapter.output_sizes)

        curve_reg = CurveRegressionHead(embed_dim=decoder_model.tok_embeddings.out_features,
                                        num_iterations=len(decoder_model.output_hidden_states) + 1,
                                        anchors=anchors_path)

        self.nn = nn.ModuleDict({'encoder': encoder_model,
                                 'decoder': decoder_model,
                                 'adapter': adapter,
                                 'regressor': curve_reg})

        self.ready_for_generation = False

    def setup_caches(self,
                     batch_size: int,
                     dtype: torch.dtype,
                     *,
                     encoder_max_seq_len: int = None,
                     decoder_max_seq_len: int = None):
        """
        Sets up key-value attention caches for inference for ``self.decoder``.
        For each layer in ``self.decoder.layers``:
        - :class:`party.modules.TransformerSelfAttentionLayer` will use ``decoder_max_seq_len``.
        - :class:`party.modules.TransformerCrossAttentionLayer` will use ``encoder_max_seq_len``.
        - :class:`party.modules.fusion.FusionLayer` will use both ``decoder_max_seq_len`` and ``encoder_max_seq_len``.

        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            encoder_max_seq_len (int): maximum encoder cache sequence length.
            decoder_max_seq_len (int): maximum decoder cache sequence length.
        """
        self.nn['decoder'].setup_caches(batch_size,
                                        dtype,
                                        encoder_max_seq_len=encoder_max_seq_len,
                                        decoder_max_seq_len=decoder_max_seq_len)

    def caches_are_setup(self) -> bool:
        """
        Check if the key value caches are setup. This means ``setup_caches`` has been called, and
        the relevant attention modules in the model have created their ``KVCache``.
        """
        return self.nn['decoder'].caches_are_setup()

    def caches_are_enabled(self) -> bool:
        """
        Checks if the key value caches are enabled. Once KV-caches have been setup, the relevant
        attention modules will be "enabled" and all forward passes will update the caches. This behaviour
        can be disabled without altering the state of the KV-caches by "disabling" the KV-caches
        using :func:`~torchtune.modules.common_utils.disable_kv_cache`, upon which ``caches_are_enabled`` would return False.
        """
        return self.nn['decoder'].caches_are_enabled()

    def reset_caches(self):
        """
        Resets KV-cache buffers on relevant attention modules to zero, and reset cache positions to zero,
        without deleting or reallocating cache tensors.
        """
        self.nn['decoder'].reset_caches()

    def forward(self,
                tokens: torch.Tensor,
                *,
                encoder_input: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input: Optional input for the encoder.
            encoder_hidden_states: Optional encoder embeddings with curve
                                   embeddings already added.
            encoder_curves: Optional curves to be embedded and added to encoder
                            embeddings.
            input_pos: Optional tensor which contains the position ids of each
                       token. During training, this is used to indicate the
                       positions of each token relative to its sample when
                       packed, shape ``[b x s]``.  During inference, this
                       indicates the position of the current token.  If none,
                       assume the index of the token is its position id.
                       Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape ``[b x s x v]`` or a list of layer \
                output tensors defined by ``output_hidden_states`` with the \
                final output tensor appended to the list.

        Notation used for tensor shapes:
            - b: batch size
            - s: token sequence length
            - s_e: encoder sequence length
            - v: vocab size
            - d: token embed dim
            - d_e: encoder embed dim
            - m_s: max seq len
        """
        # During decoding, encoder_input will only be provided
        # for new inputs. Previous encoder outputs are cached
        # in the decoder cache.
        if encoder_input is not None:
            encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)

        output = self.nn['decoder'](tokens=tokens,
                                    mask=mask,
                                    encoder_input=encoder_hidden_states,
                                    encoder_mask=encoder_mask,
                                    input_pos=input_pos)
        return self.nn['regressor'](output)

    def forward_encoder_embeddings(self, encoder_input):
        """
        Computes the encoder embeddings *without* adding the curve positional
        embeddings.
        """
        encoder_hidden_states = self.nn['encoder'](encoder_input)
        return self.nn['adapter'](encoder_hidden_states)

    def prepare_for_inference(self, config: 'Config'):
        """
        Configures the model for inference.
        """
        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        self.eval()
        self._inf_config = config

        self._max_generated_tokens = 768
        self._max_encoder_seq_len = 56800

        # set up caches
        self.setup_caches(batch_size=self._inf_config.batch_size,
                          encoder_max_seq_len=max_encoder_seq_len,
                          decoder_max_seq_len=self._max_generated_tokens,
                          dtype=next(self.encoder.parameters()).dtype)

        self.m_dtype = next(self.parameters()).dtype

        # create batch size number of BOS tokens

        self._fabric = Fabric(accelerator=self._inf_config.accelerator,
                              devices=self._inf_config.device,
                              precision=self._inf_config.precision)

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

        self._prompt = self._fabric.to_device(F.one_hot(self.bos_id, num_classes=12).unsqueeze(0).repeat(self._inf_config.batch_size, 1, 1))
        # generate a regular causal mask
        self._masks = self._fabric.to_device(torch.tril(torch.ones(self._max_generated_tokens,
                                                                   self._max_generated_tokens,
                                                                   dtype=torch.bool).unsqueeze(0)))
        self._input_pos = self._fabric.to_device(torch.arange(0, self._max_generated_tokens).unsqueeze(0))

        self.im_transforms = get_default_transforms(self.user_metadata['image_size'], dtype=self.m_dtype)

        self.ready_for_generation = True

    @torch.inference_mode()
    def predict(self, im: 'Image.Image') -> 'Segmentation':
        with self._fabric.init_tensor():
            image_input = self.im_transforms(im).unsqueeze(0)
            return self.predict_curves(image_input)

    @torch.inference_mode()
    def predict_curves(self,
                encoder_input: torch.FloatTensor) -> Generator[torch.Tensor, None, None]:
        """
        Predicts Bézier curves and line classes.

        Args:
            encoder_input: Image input for the encoder with shape ``n x c x h x w``

        Returns:
            A float tensor of with shape ``n x s x 8`` where ``n`` is the batch
            size, ``s`` is the maximum number of curves detected, and the last
            dimension is an 8-tuple defining the control points of a cubic
            Bézier curve. No-output is indicated by zeroed output control
            points. If s < max_generated_tokens the last entry across all batch
            items will be a no-output.
        """
        logger.info('Computing encoder embeddings')

        encoder_hidden_states = self.forward_encoder_embeddings(encoder_input)

        eos_token = torch.tensor(self.eos_id, device=encoder_hidden_states.device, dtype=torch.long)

        # Mask is shape (batch_size, max_seq_len, image_embedding_len)
        encoder_mask = torch.ones((encoder_hidden_states.size(0),
                                   1,
                                   encoder_hidden_states.size(1)),
                                  dtype=torch.bool,
                                  device=encoder_input.device)
        # prefill step
        curr_masks = self._masks[:, :1]
        logits = self.forward(tokens=self._prompt,
                              encoder_hidden_states=encoder_hidden_states,
                              encoder_mask=encoder_mask,
                              mask=curr_masks,
                              input_pos=self._input_pos[:, :1].squeeze())

        token_logits = logits['tokens'][-1, :, -1, :]
        tokens = torch.argmax(token_logits, dim=-1).unsqueeze(-1)
        curves = logits['curves'][-1, :, -1, :]
        generated_curves = [curves]

        curr_pos = 1

        # keeps track of EOS tokens emitted by each sequence in a batch
        eos_token_reached = torch.zeros(self._batch_size, dtype=torch.bool, device=encoder_input.device)
        eos_token_reached |= tokens[:, -1] == eos_token

        # mask used for setting all values from EOS token to pad_id in output sequences.
        eos_token_mask = torch.ones(self._batch_size, 0, dtype=torch.int32, device=curves.device)

        for _ in range(self._max_generated_tokens - 1):
            # update eos_token_mask if an EOS token was emitted in a previous step
            eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(self._batch_size, 1)], dim=-1)

            curr_input_pos = self._input_pos[:, curr_pos]
            curr_masks = self._masks[:, curr_pos, None, :]

            # no need for encoder embeddings anymore as they're in the cache now
            num_cls = logits['tokens'].shape[-1]
            input_tok = F.one_hot(tokens.squeeze(-1), num_classes=num_cls).to(device=encoder_hidden_states.device)
            logits = self.forward(tokens=torch.cat([input_tok.unsqueeze(1),
                                                    curves.unsqueeze(1)], dim=-1),
                                  mask=curr_masks,
                                  input_pos=curr_input_pos)

            token_logits = logits['tokens'][-1, :, -1, :]
            tokens = torch.argmax(token_logits, dim=-1).unsqueeze(-1)
            curves = logits['curves'][-1, :, -1, :]
            generated_curves.append(curves)

            curr_pos += 1

            eos_token_reached |= tokens[:, -1] == eos_token

            logger.info(f'Generated tokens: {tokens} Generated curves: {curves}')

            if eos_token_reached.all():
                break

        eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(self._batch_size, 1)], dim=-1)
        eos_curve_mask = eos_token_mask.expand(8, -1, -1).permute(1, 2, 0)
        return torch.stack(generated_curves, dim=1) * eos_curve_mask
