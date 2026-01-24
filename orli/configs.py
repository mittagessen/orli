from kraken.configs import TrainingConfig, SegmentationTrainingDataConfig


class OrliSegmentationTrainingConfig(TrainingConfig):
    """
    Base configuration for training a D-FINE segmentation model.

    Args:
    """
    def __init__(self, **kwargs):
        self.freeze_encoder = kwargs.pop('freeze_encoder', False)

        kwargs.setdefault('quit', 'fixed')
        kwargs.setdefault('epochs', 16)
        kwargs.setdefault('lrate', 5e-5)
        kwargs.setdefault('weight_decay', 1e-5)
        kwargs.setdefault('schedule', 'cosine')
        kwargs.setdefault('cos_max_t', 16)
        kwargs.setdefault('cos_min_lr', 5e-6)
        kwargs.setdefault('warmup', 1000)
        kwargs.setdefault('accumulate_grad_batches', 8)

        super().__init__(**kwargs)


class OrliSegmentationTrainingDataConfig(SegmentationTrainingDataConfig):
    """
    Base data configuration for a D-FINE segmentation model.
    """
    def __init__(self, **kwargs):
        self.val_batch_size = kwargs.pop('val_batch_size', None)
        self.image_size = kwargs.pop('image_size', (1024, 768))

        kwargs['line_class_mapping'] = {'DefaultLine': 1}
        kwargs['region_class_mapping'] = {} # no support for region detection
        kwargs.setdefault('batch_size', 8)
        kwargs.setdefault('format_type', 'binary')
        super().__init__(**kwargs)
