import json
import torch
import torch.serialization

from kraken.configs import TrainingConfig, SegmentationTrainingDataConfig, SegmentationInferenceConfig

from importlib.resources import files

with files('orli.assets').joinpath('anchors.json').open('r') as _fp:
    _default_anchors = tuple(tuple(row) for row in json.load(_fp))

MODEL_VARIANTS = {
    'pico': {
        'encoder_name': 'convnextv2_pico',
        'encoder_idxs': (1, 2),
        'neck_num_layers': 1,
        'neck_num_heads': 4,
        'neck_hidden_dim': 192,
        'neck_use_encoder_idx': (1,),
        'neck_output_ds_factors': (1, 2),
        'neck_norm': 'group',
        'neck_ffn_dim': 768,
        'neck_dropout': 0.0,
        'neck_fusion_depth': 2,
    },
    'tiny': {
        'encoder_name': 'convnextv2_tiny',
        'encoder_idxs': (1, 2, 3),
        'neck_num_layers': 1,
        'neck_num_heads': 8,
        'neck_hidden_dim': 256,
        'neck_use_encoder_idx': (2,),
        'neck_output_ds_factors': (1, 2, 2),
        'neck_norm': 'group',
        'neck_ffn_dim': 1024,
        'neck_dropout': 0.0,
        'neck_fusion_depth': 2,
    },
    'small': {
        'encoder_name': 'convnextv2_small',
        'encoder_idxs': (1, 2, 3),
        'neck_num_layers': 2,
        'neck_num_heads': 8,
        'neck_hidden_dim': 384,
        'neck_use_encoder_idx': (2,),
        'neck_output_ds_factors': (1, 2, 2),
        'neck_norm': 'group',
        'neck_ffn_dim': 1536,
        'neck_dropout': 0.1,
        'neck_fusion_depth': 3,
    },
}

_DEFAULT_MODEL_VARIANT = 'tiny'

class OrliSegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a D-FINE segmentation model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        anchors = kwargs.pop('anchors', _default_anchors)
        self.anchors = anchors
        self.model_variant = kwargs.pop('model_variant', _DEFAULT_MODEL_VARIANT)
        if self.model_variant not in MODEL_VARIANTS:
            choices = ', '.join(MODEL_VARIANTS)
            raise ValueError(f'Unknown model_variant {self.model_variant!r}. Choices: {choices}')
        self.slurm = kwargs.pop('slurm', False)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 16)
        kwargs.setdefault('lrate', 1e-4)
        kwargs.setdefault('weight_decay', 1e-4)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_t_max', 16)
        kwargs.setdefault('cos_min_lr', 1e-5)
        kwargs.setdefault('warmup', 2000)
        kwargs.setdefault('accumulate_grad_batches', 8)

        super().__init__(**kwargs)


class OrliSegmentationTrainingDataConfig(SegmentationTrainingDataConfig):
    """
    Base data configuration for a Orli segmentation model.
    """
    def __init__(self, **kwargs):
        self.val_batch_size = kwargs.pop('val_batch_size', None)
        self.image_size = kwargs.pop('image_size', (1280, 960))

        kwargs['line_class_mapping'] = {'DefaultLine': 1}
        kwargs['region_class_mapping'] = {}  # no support for region detection
        kwargs.setdefault('batch_size', 8)
        kwargs.setdefault('format_type', 'binary')
        super().__init__(**kwargs)


class OrliSegmentationInferenceConfig(SegmentationInferenceConfig):
    """
    Base data configuration for a Orli segmentation model.
    """
    def __init__(self, **kwargs):
        self.max_predicted_lines = kwargs.pop('max_predicted_lines', 768)
        self.polygonize = kwargs.pop('polygonize', False)
        super().__init__(**kwargs)


class OrliSegmentationTestConfig(OrliSegmentationInferenceConfig):
    """
    Configuration for Orli segmentation model testing.
    """
    def __init__(self, **kwargs):
        self.tolerance = kwargs.pop('tolerance', 10.0)
        self.match_threshold = kwargs.pop('match_threshold', 0.5)

        super().__init__(**kwargs)

torch.serialization.add_safe_globals([OrliSegmentationTrainingConfig,
                                      OrliSegmentationTrainingDataConfig,
                                      OrliSegmentationInferenceConfig,
                                      OrliSegmentationTestConfig])
