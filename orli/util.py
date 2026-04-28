# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#           2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model format conversion routines"""
import json
import torch

from safetensors.torch import save_file, _remove_duplicate_names
from typing import Optional, Union, TYPE_CHECKING

from orli.modules.baseline import DEFAULT_NUM_BASELINE_POINTS, baseline_param_dim

if TYPE_CHECKING:
    from os import PathLike


def checkpoint_to_kraken(checkpoint_path: Union[str, 'PathLike'],
                         filename: Union[str, 'PathLike'],
                         model_card: Optional[str] = None):
    """
    Converts a lightning checkpoint and optional HTRMoPo model card to the new
    safetensors-based kraken serialization format.
    """
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    # we do not have configurable encoders/decoders
    config = {"baseline_num_points": DEFAULT_NUM_BASELINE_POINTS,
              "decoder_vocab_size": 4 + baseline_param_dim(DEFAULT_NUM_BASELINE_POINTS),
              "decoder_num_layers": 4,
              "decoder_num_heads": 9,
              "decoder_num_kv_heads": 3,
              "decoder_embed_dim": 576,
              "decoder_max_seq_len": 768,
              "decoder_intermediate_dim": 1536,
              "decoder_attn_dropout": 0.0,
              "decoder_norm_eps": 1e-05,
              "decoder_rope_base": 10000,
              "encoder_name": state_dict['hyper_parameters']['encoder'],
              "encoder_input_size": state_dict['hyper_parameters']['encoder_input_size'],
              'encoder_idxs': state_dict['hyper_parameters']['encoder_idxs']}
    model_type = 'kraken_orli'
    metadata = {'model_type': model_type,
                'config': json.dumps(config)}
    if model_card:
        metadata['model_card'] = model_card

    state_dict = {k.removeprefix('model.'): v for k, v in state_dict['state_dict'].items()}

    to_removes = _remove_duplicate_names(state_dict)

    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                # Do not override user data
                metadata[to_remove] = kept_name
            del state_dict[to_remove]

    # we can just save the state dict as our constructor sets up the tensor
    # sharing.
    save_file(state_dict, filename, metadata=metadata)


def update_model_card(model_path: Union[str, 'PathLike'],
                      model_card: Optional[str] = None):
    """
    Replaces the current model card inside a model in the safetensors-based
    kraken serialization format with a new one.
    """
    pass
