import logging
from pathlib import Path
from typing import Union

from .types import TrainLogger

log = logging.getLogger(__name__)


class TensorBoardLogger(TrainLogger):
    def __init__(
        self,
        use_ortho_loss=False,
        class_specific=True,
        calculate_best_for=["accu"],
        device="cpu",
    ):
        super().__init__(
            use_ortho_loss=use_ortho_loss,
            calculate_best_for=calculate_best_for,
            device=device,
        )

        # TODO: Remove this when metrics are unhardcoded
        self.class_specific = class_specific

    def log_metrics(
        self,
        is_train,
        precalculated_metrics=None,
        prototype_embedded_state=False,
        step=None,
    ):
        metrics = self.train_metrics if is_train else self.val_metrics
        tag = "train" if is_train else "validation"

        # Log the computed metric values
        for name, metric in metrics.items():
            if metric._update_called:
                computed_value = metric.compute()
                log.info(f"{name} ({tag}): {computed_value}")
                metric.reset()  # Reset after logging for the next epoch

        # TODO: Unify with other metrics
        if precalculated_metrics:
            for name, value in precalculated_metrics.items():
                log.info(f"{name}: {value}")

    # FIXME - this should be handled by the parent class
    def end_epoch(
        self,
        epoch_metrics_dict,
        is_train,
        epoch_index,
        prototype_embedded_epoch,
        precalculated_metrics=None,
    ):
        if self.use_ortho_loss:
            log.info("\t Using ortho loss")

        for key in epoch_metrics_dict:
            # DO NOTHING FOR THESE KEYS
            if (
                key
                not in [
                    "time",
                    "n_batches",
                    "l1",
                    "max_offset",
                    "n_correct",
                    "n_examples",
                    "accu",
                    "weighted_auroc",
                    "is_train",
                ]
                and epoch_metrics_dict[key]
            ):
                epoch_metrics_dict[key] /= epoch_metrics_dict["n_batches"]

        self.update_metrics(epoch_metrics_dict, is_train)

        complete_metrics = epoch_metrics_dict.copy()
        if precalculated_metrics is not None:
            complete_metrics.update(precalculated_metrics)

        self.update_bests(
            complete_metrics,
            step=epoch_index,
            is_train=is_train,
            prototype_embedded_epoch=prototype_embedded_epoch,
        )
        self.log_metrics(
            is_train,
            prototype_embedded_state=prototype_embedded_epoch,
            precalculated_metrics=precalculated_metrics,
            step=epoch_index,
        )

        for key in epoch_metrics_dict:
            # if class specific is true, print separation and avg_separation
            # always print the rest
            if self.class_specific or key not in ["separation", "avg_separation"]:
                log.info(
                    "\t{0}: \t{1}".format(key, epoch_metrics_dict[key]),
                )

    def log_best_model(self, model_path: Union[str, Path]):
        log.info("New best model saved to %s", Path(model_path).absolute())

    @staticmethod
    def log_backdrops(backdrop_dict, step=None):
        lr_str = "|".join(
            f"{name}:{value:.3g}" for name, value in backdrop_dict.items()
        )
        log.info("learning rates are %s", lr_str)
