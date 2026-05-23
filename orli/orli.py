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

from orli.configs import OrliSegmentationInferenceConfig, MODEL_VARIANTS
from orli.fusion import baseline_decoder, OrliHybridNeck, CurveRegressionHead
from orli.dataset import get_default_transforms
from orli.modules.baseline import (DEFAULT_NUM_BASELINE_POINTS,
                                   curve_vector_dim,
                                   curve_vector_to_polyline)

from kraken.models import SegmentationBaseModel
from kraken.containers import Segmentation, BaselineLine

import uuid

logger = logging.getLogger(__name__)


__all__ = ['OrliModel']


class OrliModel(nn.Module, SegmentationBaseModel):
    """
    The transformer segmentation fusion model.

    Args:
    """
    bos_id: int = 1
    eos_id: int = 2
    model_type = ['segmentation']
    _kraken_min_version = '6.99.99'

    def __init__(self, **kwargs):
        super().__init__()

        if (config := kwargs.get('config', None)) is None:
            raise ValueError('config argument is missing in args.')
        if hasattr(config, '__dict__'):
            config = {'anchors': config.anchors,
                      'model_variant': config.model_variant,
                      'baseline_num_points': getattr(config,
                                                     'baseline_num_points',
                                                     DEFAULT_NUM_BASELINE_POINTS),
                      'curve_fourier_features': getattr(config,
                                                        'curve_fourier_features',
                                                        True),
                      'anchor_embedding': getattr(config,
                                                  'anchor_embedding',
                                                  True),
                      'line_refiner': getattr(config,
                                              'line_refiner',
                                              False),
                      'direct_point_regression': getattr(config,
                                                         'direct_point_regression',
                                                         False),
                      'curve_prompt_noise_prob': getattr(config,
                                                         'curve_prompt_noise_prob',
                                                         0.0),
                      'curve_prompt_noise_normal_px': getattr(config,
                                                              'curve_prompt_noise_normal_px',
                                                              0.0),
                      'curve_prompt_noise_tangent_px': getattr(config,
                                                               'curve_prompt_noise_tangent_px',
                                                               0.0),
                      'curve_prompt_noise_curvature_px': getattr(config,
                                                                 'curve_prompt_noise_curvature_px',
                                                                 0.0),
                      'pre_refiner_noise_prob': getattr(config,
                                                        'pre_refiner_noise_prob',
                                                        0.0),
                      'pre_refiner_noise_normal_px': getattr(config,
                                                             'pre_refiner_noise_normal_px',
                                                             0.0),
                      'pre_refiner_noise_tangent_px': getattr(config,
                                                              'pre_refiner_noise_tangent_px',
                                                              0.0),
                      'pre_refiner_noise_curvature_px': getattr(config,
                                                                'pre_refiner_noise_curvature_px',
                                                                0.0)}
        config = dict(config)
        config.pop('soft_anchors', None)
        if config.get('baseline_num_points') is None:
            config['baseline_num_points'] = DEFAULT_NUM_BASELINE_POINTS
        if config.get('curve_fourier_features') is None:
            config['curve_fourier_features'] = True
        if config.get('anchor_embedding') is None:
            config['anchor_embedding'] = True
        if config.get('line_refiner') is None:
            config['line_refiner'] = False
        if config.get('direct_point_regression') is None:
            config['direct_point_regression'] = False
        for key in ('curve_prompt_noise_prob',
                    'curve_prompt_noise_normal_px',
                    'curve_prompt_noise_tangent_px',
                    'curve_prompt_noise_curvature_px',
                    'pre_refiner_noise_prob',
                    'pre_refiner_noise_normal_px',
                    'pre_refiner_noise_tangent_px',
                    'pre_refiner_noise_curvature_px'):
            if config.get(key) is None:
                config[key] = 0.0

        model_variant = config.get('model_variant', 'tiny')
        if model_variant not in MODEL_VARIANTS:
            choices = ', '.join(MODEL_VARIANTS)
            raise ValueError(f'Unknown model_variant {model_variant!r}. Choices: {choices}')
        metadata = dict(kwargs)
        metadata['config'] = config
        kwargs = {**kwargs, **config, **MODEL_VARIANTS[model_variant]}

        if (image_size := kwargs.get('image_size', None)) is None:
            raise ValueError('image_size argument is missing in args.')

        if (anchors := config.get('anchors', None)) is None:
            raise ValueError('anchors argument is missing in config.')
        self.baseline_num_points = int(config.get('baseline_num_points',
                                                  DEFAULT_NUM_BASELINE_POINTS))
        self.direct_point_regression = bool(config.get('direct_point_regression', False))
        self.baseline_dim = curve_vector_dim(self.baseline_num_points,
                                             self.direct_point_regression)

        encoder_name = kwargs['encoder_name']
        encoder_idxs = list(kwargs['encoder_idxs'])
        neck_num_layers = kwargs['neck_num_layers']
        neck_num_heads = kwargs['neck_num_heads']
        neck_hidden_dim = kwargs['neck_hidden_dim']
        neck_use_encoder_idx = kwargs['neck_use_encoder_idx']
        neck_output_ds_factors = kwargs['neck_output_ds_factors']
        neck_norm = kwargs['neck_norm']
        neck_ffn_dim = kwargs['neck_ffn_dim']
        neck_dropout = kwargs['neck_dropout']
        neck_fusion_depth = kwargs['neck_fusion_depth']

        self.user_metadata: dict[str, Any] = {'model_type': self.model_type,
                                               'accuracy': [],
                                               'metrics': []}
        self.user_metadata.update(metadata)

        logger.info('Creating segmentation model')

        encoder_model = timm.create_model(encoder_name,
                                          pretrained=True,
                                          features_only=True,
                                          out_indices=encoder_idxs)

        encoder_sizes = [(int(image_size[0]/encoder_model.feature_info.reduction(idx)),
                          int(image_size[1]/encoder_model.feature_info.reduction(idx)),
                          encoder_model.feature_info.channels(idx)) for idx in encoder_idxs]

        adapter = OrliHybridNeck(encoder_embed_dims=[x[2] for x in encoder_sizes],
                                 encoder_sizes=[x[:2] for x in encoder_sizes],
                                 decoder_embed_dim=576,
                                 hidden_dim=neck_hidden_dim,
                                 num_heads=neck_num_heads,
                                 num_encoder_layers=neck_num_layers,
                                 use_encoder_idx=neck_use_encoder_idx,
                                 output_ds_factors=neck_output_ds_factors,
                                 norm_type=neck_norm,
                                 dim_feedforward=neck_ffn_dim,
                                 dropout=neck_dropout,
                                 fusion_depth=neck_fusion_depth)

        decoder_model = baseline_decoder(vocab_size=4 + self.baseline_dim,
                                         encoder_sizes=adapter.output_sizes,
                                         curve_num_freqs=4 if config.get('curve_fourier_features', True) else 0)

        curve_reg = CurveRegressionHead(embed_dim=decoder_model.tok_embeddings.embed_dim,
                                        num_iterations=len(decoder_model.output_hidden_states) + 1,
                                        num_baseline_points=self.baseline_num_points,
                                        direct_point_regression=self.direct_point_regression,
                                        anchor_embedding=config.get('anchor_embedding', True),
                                        line_refiner=config.get('line_refiner', False),
                                        image_size=(image_size[1], image_size[0]),
                                        curve_prompt_noise_prob=config.get('curve_prompt_noise_prob', 0.0),
                                        curve_prompt_noise_normal_px=config.get('curve_prompt_noise_normal_px', 0.0),
                                        curve_prompt_noise_tangent_px=config.get('curve_prompt_noise_tangent_px', 0.0),
                                        curve_prompt_noise_curvature_px=config.get('curve_prompt_noise_curvature_px',
                                                                                   0.0),
                                        pre_refiner_noise_prob=config.get('pre_refiner_noise_prob', 0.0),
                                        pre_refiner_noise_normal_px=config.get('pre_refiner_noise_normal_px', 0.0),
                                        pre_refiner_noise_tangent_px=config.get('pre_refiner_noise_tangent_px', 0.0),
                                        pre_refiner_noise_curvature_px=config.get('pre_refiner_noise_curvature_px',
                                                                                  0.0),
                                        anchors=anchors)

        self.nn = nn.ModuleDict({'encoder': encoder_model,
                                 'decoder': decoder_model,
                                 'adapter': adapter,
                                 'regressor': curve_reg})

        self.ready_for_generation = False

    @property
    def user_metadata(self) -> dict[str, Any]:
        return self._user_metadata

    @user_metadata.setter
    def user_metadata(self, val: dict[str, Any]) -> None:
        self._user_metadata = val

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
                refiner_encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_curves: Optional[torch.Tensor] = None,
                target_anchor_idx: Optional[torch.Tensor] = None,
                encoder_mask: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                input_pos: Optional[torch.Tensor] = None) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            encoder_input: Optional input for the encoder.
            encoder_hidden_states: Optional encoder embeddings with curve
                                   embeddings already added.
            refiner_encoder_hidden_states: Optional encoder embeddings used
                                           only by the local curve refiner.
            encoder_curves: Optional curves to be embedded and added to encoder
                            embeddings.
            target_anchor_idx: Optional per-token anchor assignments used during
                               training to initialize multi-anchor regression.
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

        refiner_memory = (refiner_encoder_hidden_states
                          if refiner_encoder_hidden_states is not None
                          else encoder_hidden_states)
        output = self.nn['decoder'](tokens=tokens,
                                    mask=mask,
                                    encoder_input=encoder_hidden_states,
                                    encoder_mask=encoder_mask,
                                    input_pos=input_pos)
        return self.nn['regressor'](output,
                                    target_anchor_idx=target_anchor_idx,
                                    encoder_hidden_states=refiner_memory,
                                    encoder_sizes=self.nn['adapter'].output_sizes)

    def forward_encoder_embeddings(self, encoder_input):
        """
        Computes the encoder embeddings *without* adding the curve positional
        embeddings.
        """
        encoder_hidden_states = self.nn['encoder'](encoder_input)
        return self.nn['adapter'](encoder_hidden_states)

    def prepare_for_inference(self, config):
        """
        Configures the model for inference.

        Args:
            config: An inference configuration object. Works with both
                    :class:`OrliSegmentationInferenceConfig` and the base
                    :class:`SegmentationInferenceConfig` (orli-specific
                    options fall back to defaults in the latter case).
        """
        if self.ready_for_generation:
            logger.debug('Model has already been prepared for generation!')

        self.eval()
        self._inf_config = config

        max_predicted_lines = getattr(config, 'max_predicted_lines', 768)

        self._fabric = Fabric(accelerator=config.accelerator,
                              devices=config.device,
                              precision=config.precision)

        self.nn = self._fabric._precision.convert_module(self.nn)
        self.nn = self._fabric.to_device(self.nn)

        self.m_dtype = next(self.parameters()).dtype

        self._max_encoder_seq_len = sum(h * w for h, w in self.nn['adapter'].output_sizes)

        self._resize_generation_state(config.batch_size)

        # generate a regular causal mask
        self._masks = self._fabric.to_device(torch.tril(torch.ones(max_predicted_lines,
                                                                   max_predicted_lines,
                                                                   dtype=torch.bool).unsqueeze(0)))
        self._input_pos = self._fabric.to_device(torch.arange(0, max_predicted_lines).unsqueeze(0))

        self.im_transforms = get_default_transforms(self.user_metadata['image_size'], dtype=self.m_dtype)

        self.ready_for_generation = True

    def _resize_generation_state(self, batch_size: int) -> None:
        max_predicted_lines = getattr(self._inf_config, 'max_predicted_lines', 768)
        self._batch_size = batch_size
        self.setup_caches(batch_size=self._batch_size,
                          encoder_max_seq_len=self._max_encoder_seq_len,
                          decoder_max_seq_len=max_predicted_lines,
                          dtype=self.m_dtype)

        bos = torch.zeros(4 + self.baseline_dim, dtype=self.m_dtype)
        bos[self.bos_id] = 1.0
        self._prompt = self._fabric.to_device(bos.unsqueeze(0).unsqueeze(0).repeat(self._batch_size, 1, 1))

    @torch.inference_mode()
    def predict(self, im: 'Image.Image') -> 'Segmentation':
        if not self.ready_for_generation:
            raise RuntimeError('Model must be prepared for inference first. Call prepare_for_inference().')

        with self._fabric.init_tensor():
            image_input = self.im_transforms(im).unsqueeze(0)
        image_input = self._fabric.to_device(image_input)

        curves = self.predict_curves(image_input)
        if curves.dim() == 3:
            curves = curves[0]
        non_empty = curves.abs().sum(dim=-1) > 0
        curves = curves[non_empty]

        sampled = curve_vector_to_polyline(curves,
                                           direct_point_regression=self.direct_point_regression,
                                           num_points=self.baseline_num_points).clamp(0.0, 1.0)
        if sampled.numel():
            scale = torch.tensor((im.width, im.height),
                                 dtype=sampled.dtype,
                                 device=sampled.device)
            sampled = sampled * scale
        sampled = sampled.cpu()

        lines = []
        baselines = []
        for line in sampled:
            baseline = torch.round(line).to(torch.int64).tolist()
            baseline = tuple(map(tuple, baseline))
            baselines.append(baseline)
            lines.append(BaselineLine(id=f'_{uuid.uuid4()}',
                                      baseline=baseline,
                                      tags={'type': [{'type': 'default'}]}))

        if getattr(self._inf_config, 'polygonize', False) and lines:
            from kraken.lib.segmentation import calculate_polygonal_environment
            boundaries = calculate_polygonal_environment(im=im.convert('L'),
                                                         baselines=[list(bl) for bl in baselines])
            for line, boundary in zip(lines, boundaries):
                if boundary is not None:
                    line.boundary = boundary

        return Segmentation(type='baselines',
                            imagename=getattr(im, 'filename', '') or '',
                            script_detection=False,
                            text_direction=self._inf_config.text_direction,
                            lines=lines)

    @torch.inference_mode()
    def predict_curves(self, encoder_input: torch.FloatTensor) -> Generator[torch.Tensor, None, None]:
        """
        Predicts local-frame baseline vectors and line classes.

        Args:
            encoder_input: Image input for the encoder with shape ``n x c x h x w``

        Returns:
            A float tensor of with shape ``n x s x d`` where ``n`` is the batch
            size, ``s`` is the maximum number of baselines detected, and ``d``
            is the local-frame baseline vector width. No-output is indicated by
            zeroed output vectors. If s < max_generated_tokens the last entry
            across all batch items will be a no-output.
        """
        logger.info('Computing encoder embeddings')

        batch_size = encoder_input.size(0)
        if batch_size != self._batch_size:
            self._resize_generation_state(batch_size)
        elif self.caches_are_setup():
            self.reset_caches()

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

        for _ in range(getattr(self._inf_config, 'max_predicted_lines', 768) - 1):
            # update eos_token_mask if an EOS token was emitted in a previous step
            eos_token_mask = torch.cat([eos_token_mask, ~eos_token_reached.reshape(self._batch_size, 1)], dim=-1)

            curr_input_pos = self._input_pos[:, curr_pos]
            curr_masks = self._masks[:, curr_pos, None, :]

            # no need for encoder embeddings anymore as they're in the cache now
            input_tok = F.one_hot(tokens.squeeze(-1), num_classes=4).to(device=encoder_hidden_states.device)
            logits = self.forward(tokens=torch.cat([input_tok.unsqueeze(1),
                                                    curves.unsqueeze(1)], dim=-1),
                                  refiner_encoder_hidden_states=encoder_hidden_states,
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
        eos_curve_mask = eos_token_mask.expand(self.baseline_dim, -1, -1).permute(1, 2, 0)
        self._last_eos_reached = eos_token_reached.detach().clone()
        return torch.stack(generated_curves, dim=1) * eos_curve_mask
