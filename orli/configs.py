import json
import torch
import torch.serialization

from kraken.configs import TrainingConfig, SegmentationTrainingDataConfig, SegmentationInferenceConfig

from importlib.resources import files

with files('orli.assets').joinpath('anchors.json').open('r') as _fp:
    _default_anchors = tuple(tuple(row) for row in json.load(_fp))

_MODEL_ONLY_KEYS = {'freeze_encoder',
                    'anchors',
                    'encoder_name',
                    'encoder_idxs',
                    'neck_type',
                    'neck_num_layers',
                    'neck_num_heads',
                    'neck_hidden_dim',
                    'neck_use_encoder_idx',
                    'neck_output_ds_factors',
                    'neck_norm',
                    'neck_ffn_dim',
                    'neck_dropout',
                    'neck_fusion_depth',
                    'teacher_force_anchors',
                    'fourier_features',
                    'logit_refinement',
                    'slurm'}


def _strip_model_only_kwargs(kwargs):
    for key in _MODEL_ONLY_KEYS:
        kwargs.pop(key, None)


class OrliSegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a D-FINE segmentation model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        anchors = kwargs.pop('anchors', _default_anchors)
        self.anchors = anchors
        self.encoder_name = kwargs.pop('encoder_name', 'convnextv2_tiny')
        self.encoder_idxs = tuple(kwargs.pop('encoder_idxs', (1, 2, 3)))
        self.neck_type = kwargs.pop('neck_type', 'simple')
        self.neck_num_layers = kwargs.pop('neck_num_layers', 1)
        self.neck_num_heads = kwargs.pop('neck_num_heads', 8)
        self.neck_hidden_dim = kwargs.pop('neck_hidden_dim', 256)
        neck_use_encoder_idx = kwargs.pop('neck_use_encoder_idx', None)
        self.neck_use_encoder_idx = None if neck_use_encoder_idx is None else tuple(neck_use_encoder_idx)
        neck_output_ds_factors = kwargs.pop('neck_output_ds_factors', None)
        self.neck_output_ds_factors = None if neck_output_ds_factors is None else tuple(neck_output_ds_factors)
        self.neck_norm = kwargs.pop('neck_norm', 'group')
        self.neck_ffn_dim = kwargs.pop('neck_ffn_dim', 1024)
        self.neck_dropout = kwargs.pop('neck_dropout', 0.0)
        self.neck_fusion_depth = kwargs.pop('neck_fusion_depth', 2)
        # Validation-only switch. Training always uses teacher-forced anchors.
        self.teacher_force_anchors = kwargs.pop('teacher_force_anchors', True)
        self.fourier_features = kwargs.pop('fourier_features', True)
        self.logit_refinement = kwargs.pop('logit_refinement', True)
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
        _strip_model_only_kwargs(kwargs)

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
        _strip_model_only_kwargs(kwargs)

        super().__init__(**kwargs)


class OrliSegmentationTestConfig(OrliSegmentationInferenceConfig):
    """
    Configuration for Orli segmentation model testing.
    """
    def __init__(self, **kwargs):
        self.tolerance = kwargs.pop('tolerance', 10.0)
        self.match_threshold = kwargs.pop('match_threshold', 0.5)

        super().__init__(**kwargs)


torch.serialization.add_safe_globals([OrliSegmentationTrainingConfig, OrliSegmentationTrainingConfig])
