import random
from typing import Any, Dict, Optional

import torchmetrics

from ...utilities.trainer_utilities import is_single_valued_metric


class TrainLogger:
    def __init__(
        self,
        use_ortho_loss=False,
        # FIXME: this should consistently be called accuracy
        calculate_best_for=["accu"],
        device="cpu",
    ):
        self.use_ortho_loss = use_ortho_loss
        # FIXME: this should support min and max
        calculate_best_for = (
            [calculate_best_for]
            if isinstance(calculate_best_for, str)
            else calculate_best_for
        )
        self.bests = self.__setup_bests(calculate_best_for)

        # Create separate metrics dictionaries for train and validation
        self.train_metrics = self.create_metrics(device)
        self.val_metrics = self.create_metrics(device)

    def start_run(
        self, run_id: Optional[str] = None, run_meta: Optional[Dict[str, Any]] = {}
    ) -> str:
        """
        Starts a new run and returns the run ID.
        """
        return str(random.randint(0, 100000)) if run_id is None else run_id

    # FIXME: this should be part of the metrics class, not the logger
    def __setup_bests(self, calculate_best_for):
        bests = {}
        for train_eval in ["train", "eval"]:
            bests[train_eval] = {}
            for metric_name in calculate_best_for:
                # FIXME: this should support min and max
                bests[train_eval][metric_name] = {
                    "any": float("-inf"),
                    "prototypes_embedded": float("-inf"),
                }

        return bests

    def update_bests(
        self, metrics_dict, step, prototype_embedded_epoch=False, is_train=False
    ):
        metric_group = "train" if is_train else "eval"
        for metric_name, metric_value in metrics_dict.items():
            if metric_name in self.bests[metric_group] and metric_value is not None:
                # attempt to update best
                # TODO: accommodate for multivalued metrics
                if is_single_valued_metric(metric_value):
                    if metric_value > self.bests[metric_group][metric_name]["any"]:
                        self.bests[metric_group][metric_name]["any"] = metric_value
                        self.process_new_best(
                            self.__metric_best_name(metric_group, metric_name, False),
                            metric_value,
                            step,
                        )

                    if prototype_embedded_epoch:
                        if (
                            metric_value
                            > self.bests[metric_group][metric_name][
                                "prototypes_embedded"
                            ]
                        ):
                            self.bests[metric_group][metric_name][
                                "prototypes_embedded"
                            ] = metric_value
                            self.process_new_best(
                                self.__metric_best_name(
                                    metric_group, metric_name, True
                                ),
                                metric_value,
                                step,
                            )

    def __metric_best_name(self, train_eval, metric_name, prototype_embedded_state):
        maybe_prototypes_embedded = (
            "[prototypes_embedded]" if prototype_embedded_state else ""
        )
        return f"best{maybe_prototypes_embedded}/{train_eval}/{metric_name}"

    def serialize_bests(self):
        bests_flat = {}
        for train_eval, metrics_dict in self.bests.items():
            for metric_name, metric_values in metrics_dict.items():
                bests_flat[
                    self.__metric_best_name(train_eval, metric_name, False)
                ] = metric_values["any"]
                bests_flat[
                    self.__metric_best_name(train_eval, metric_name, True)
                ] = metric_values["prototypes_embedded"]
        return bests_flat

    def process_new_best(
        self, metric_name, metric_value, step, prototype_embedded_state=False
    ):
        """
        This method is called whenever a new "best" value of a metric is found with the value of the metric, the current, step,
        and whether the prototype layer is embedded or not. It provides a hook to capture the new value and take any necessary actions.

        The default is a no-op. Subclasses can override this method to implement custom behavior.
        """
        pass

    def create_metrics(self, device):
        # Helper method to initialize metrics
        return {
            "n_examples": torchmetrics.SumMetric().to(device),
            "n_correct": torchmetrics.SumMetric().to(device),
            "n_batches": torchmetrics.SumMetric().to(device),
            "cross_entropy": torchmetrics.MeanMetric().to(device),
            "cluster": torchmetrics.MeanMetric().to(device),
            "separation": torchmetrics.MeanMetric().to(device),
            "fine_annotation": torchmetrics.MeanMetric().to(device),
            "accu": torchmetrics.MeanMetric().to(
                device
            ),  # Using torchmetrics.Accuracy directly for accuracy
            "l1": torchmetrics.MeanMetric().to(device),
            "total_loss": torchmetrics.MeanMetric().to(device),
            "nll_loss": torchmetrics.MeanMetric().to(device),
            "orthogonality_loss": torchmetrics.MeanMetric().to(device),
            "contrastive_masked": torchmetrics.MeanMetric().to(device),
            "contrastive_unmasked": torchmetrics.MeanMetric().to(device),
            "n_unique_proto_parts": torchmetrics.MeanMetric().to(device),
            "n_unique_protos": torchmetrics.MeanMetric().to(device),
            "grassmannian_orthogonality_loss": torchmetrics.MeanMetric().to(device),
        }

    def update_metrics(self, metrics_dict, is_train):
        metrics = self.train_metrics if is_train else self.val_metrics

        # Update each metric from metrics_dict
        for key, value in metrics_dict.items():
            # TODO: Is this desired- not tracking Nones (torchmetrics does not like them)
            if key in metrics and value is not None:
                metrics[key].update(value)
