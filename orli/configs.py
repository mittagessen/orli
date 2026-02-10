import torch.serialization

from kraken.configs import TrainingConfig, SegmentationTrainingDataConfig, SegmentationInferenceConfig

from importlib.resources import files


class OrliSegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a D-FINE segmentation model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)
        self.anchors_path = kwargs.pop('anchors_path', files('orli.assets').joinpath('anchors.pt'))

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
