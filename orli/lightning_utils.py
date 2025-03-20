import optuna
import warnings

from lightning.pytorch import Callback

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightning.pytorch as pl


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/lightning.pytorch_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``lightning.pytorch.LightningModule.training_step`` or
            ``lightning.pytorch.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.

    .. note::
        If you would like to use PyTorchLightningPruningCallback in a distributed training
        environment, you need to evoke `PyTorchLightningPruningCallback.check_pruned()`
        manually so that :class:`~optuna.exceptions.TrialPruned` is properly handled.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = pl_module.current_epoch
        # Determine if the trial should be terminated in a single process.
        self._trial.report(current_score.item(), step=epoch)
        if not self._trial.should_prune():
            return
        raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")
