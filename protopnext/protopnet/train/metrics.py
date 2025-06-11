import logging
from dataclasses import dataclass
from typing import List, Union

import torch
import torchmetrics
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassPrecisionRecallCurve,
    MulticlassROC,
)

from ..metrics import PartConsistencyScore, PartStabilityScore, add_gaussian_noise
from ..prototypical_part_model import ProtoPNet
from ..utilities.trainer_utilities import predicated_extend

log = logging.getLogger(__name__)


@dataclass
class TrainingMetric:
    name: str
    # min, max
    metric: torchmetrics.Metric

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, metric={self.metric})"

    def __str__(self):
        return f"{self.name}: {self.metric.compute()}"


# TODO: Make all the metrics be calculated in the same way
class TrainingMetrics:
    def __init__(
        self,
        metrics: List[TrainingMetric] = [],
        device: Union[str, torch.device] = "cpu",
    ):
        self.metrics = {metric.name: metric for metric in metrics}

        for metric in self.metrics.values():
            metric.metric.to(device)

    def start_epoch(self, phase: str):
        self.reset()

    def end_epoch(self, phase: str):
        self.reset()

    def metric_names(self):
        return list(self.metrics.keys())

    def reset(self):
        for _, metric in self.metrics.items():
            metric.metric.reset()

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        return {
            metric_name: metric.metric.compute()
            for metric_name, metric in self.metrics.items()
        }

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        if len(self.metrics) > 0:
            raise NotImplementedError("This method must be implemented in a subclass.")

        # otherwise, do nothing, which is fine


class InterpretableTrainingMetrics(TrainingMetrics):
    """
    This is a temporary implementation of training metrics that lets us easily aggregate the interpretable
    metrics during an initial training run.
    """

    def __init__(
        self,
        protopnet: ProtoPNet,
        num_classes: int,
        proto_per_class: int,
        part_num: int,
        img_size: 224,
        half_size: 36,
        device: Union[str, torch.device] = "cpu",
        acc_only: bool = False,
    ):
        """
        Args:
            protopnet (ProtoPNet): The ProtoPNet model.
            num_classes (int): The number of classes in the dataset.
            proto_per_class (int): The default number of prototypes per class.
            part_num (int): The number of parts in the dataset.
            img_size (int): The size of the input images.
            half_size (int): The size of the half of the input image. See: metrics.InterpretableMetrics
            device (Union[str, torch.device], optional): The device to run the metrics on. Defaults to "cpu".
        """

        super().__init__(
            metrics=predicated_extend(
                not acc_only,
                [
                    TrainingMetric(
                        name="accuracy",
                        metric=torchmetrics.Accuracy(
                            num_classes=num_classes,
                            task="multiclass",
                        ),
                    ),
                    TrainingMetric(
                        name="pr",
                        metric=MulticlassPrecisionRecallCurve(
                            num_classes=num_classes,
                        ),
                    ),
                    TrainingMetric(
                        name="roc",
                        metric=MulticlassROC(
                            num_classes=num_classes,
                        ),
                    ),
                    TrainingMetric(
                        name="weighted_auroc",
                        metric=MulticlassAUROC(
                            num_classes=num_classes,
                            average="weighted",
                            thresholds=None,
                        ),
                    ),
                    TrainingMetric(
                        name="class_aurocs",
                        metric=MulticlassAUROC(
                            num_classes=num_classes,
                            average=None,
                            thresholds=None,
                        ),
                    ),
                    TrainingMetric(
                        name="conf_mat",
                        metric=MulticlassConfusionMatrix(
                            num_classes=num_classes,
                        ),
                    ),
                ],
                [
                    TrainingMetric(
                        name="prototype_stability",
                        metric=PartStabilityScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_consistency",
                        metric=PartConsistencyScore(
                            num_classes=num_classes,
                            part_num=part_num,
                            proto_per_class=proto_per_class,
                            img_sz=img_size,
                            half_size=half_size,
                        ),
                    ),
                    TrainingMetric(
                        name="prototype_sparsity",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_proto_parts",
                        metric=torchmetrics.MeanMetric(),
                    ),
                    TrainingMetric(
                        name="n_unique_protos",
                        metric=torchmetrics.MeanMetric(),
                    ),
                ],
            ),
            device=device,
        )

        self.generator = torch.Generator(device=device)
        self.protopnet = protopnet

        self.prototypes_embedded_last = False
        self.prototype_metrics_cached = False
        self.prototypes_embedded_any = False
        # FIXME - this is hack
        self.acc_only = acc_only

    def metric_names(self):
        raw_metric_names = super().metric_names()
        if self.acc_only:
            return raw_metric_names
        else:
            return raw_metric_names + [
                "prototype_score",
                "acc_proto_score",
            ]

    def start_epoch(self, phase: str):
        # This assumes that project only has eval once - if eval was run on train first, it would break this
        if self.protopnet.prototypes_embedded() and not self.prototypes_embedded_last:
            self.reset()
            self.prototype_metrics_cached = False
        else:
            self.metrics["accuracy"].metric.reset()
            self.metrics["pr"].metric.reset()
            self.metrics["roc"].metric.reset()
            self.metrics["weighted_auroc"].metric.reset()
            self.metrics["class_aurocs"].metric.reset()
            self.metrics["conf_mat"].metric.reset()

        if self.protopnet.prototypes_embedded():
            self.prototypes_embedded_any = True
        self.prototypes_embedded_last = self.protopnet.prototypes_embedded()

    def update_all(self, forward_args: dict, forward_outputs: dict, phase: str):
        """
        Update all the metrics.
        """

        self.update_accuracy(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_pr(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_roc(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_weighted_auroc(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_class_aurocs(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )
        self.update_conf_mat(
            forward_args=forward_args,
            forward_outputs=forward_outputs,
        )

        if not self.acc_only and "sample_bounding_box" in forward_args:
            self.update_stability(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_consistency(
                forward_args=forward_args,
                forward_outputs=forward_outputs,
            )
            self.update_prototype_sparsity()

    def update_accuracy(self, forward_args: dict, forward_outputs: dict):
        """
        Update the accuracy metric.
        """
        accuracy = self.metrics["accuracy"].metric
        accuracy.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_pr(self, forward_args: dict, forward_outputs: dict):
        """
        Update the precision-recall metric.
        """
        pr = self.metrics["pr"].metric
        pr.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_roc(self, forward_args: dict, forward_outputs: dict):
        """
        Update the roc metric.
        """
        roc = self.metrics["roc"].metric
        roc.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_weighted_auroc(self, forward_args: dict, forward_outputs: dict):
        """
        Update the weighted auroc metric.
        """
        weighted_auroc = self.metrics["weighted_auroc"].metric
        weighted_auroc.update(
            preds=forward_outputs["logits"], target=forward_args["target"]
        )

    def update_class_aurocs(self, forward_args: dict, forward_outputs: dict):
        """
        Update the class aurocs metric.
        """
        class_aurocs = self.metrics["class_aurocs"].metric
        class_aurocs.update(
            preds=forward_outputs["logits"], target=forward_args["target"]
        )

    def update_conf_mat(self, forward_args: dict, forward_outputs: dict):
        """
        Update the confusion matrix metric.
        """
        conf_mat = self.metrics["conf_mat"].metric
        conf_mat.update(preds=forward_outputs["logits"], target=forward_args["target"])

    def update_stability(self, forward_args: dict, forward_outputs: dict):
        """
        Update the stability metric.
        """
        with torch.no_grad():
            # Compute the consistency metric (prototype_activations_noisy)
            proto_acts_noisy = self.protopnet(
                add_gaussian_noise(forward_args["img"], self.generator),
                return_prototype_layer_output_dict=True,
            )["prototype_activations"].detach()

        stability = self.metrics["prototype_stability"].metric
        stability.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            proto_acts_noisy=proto_acts_noisy,
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_consistency(self, forward_args: dict, forward_outputs: dict):
        """
        Update the consistency metric.
        """
        consistency = self.metrics["prototype_consistency"].metric

        consistency.update(
            proto_acts=forward_outputs["prototype_activations"],
            targets=forward_args["target"],
            sample_parts_centroids=forward_args["sample_parts_centroids"],
            sample_bounding_box=forward_args["sample_bounding_box"],
        )

    def update_prototype_sparsity(self):
        """
        Update the prototype sparsity metric.
        """
        prototype_complexity_stats = self.protopnet.get_prototype_complexity()

        self.metrics["prototype_sparsity"].metric.update(
            prototype_complexity_stats["prototype_sparsity"]
        )
        self.metrics["n_unique_protos"].metric.update(
            prototype_complexity_stats["n_unique_protos"]
        )
        self.metrics["n_unique_proto_parts"].metric.update(
            prototype_complexity_stats["n_unique_proto_parts"]
        )

    def compute_dict(self) -> dict:
        """
        Compute all the metrics and return the raw values in a dictionary.
        """
        if self.acc_only or not self.prototypes_embedded_any:
            return {
                "accuracy": self.metrics["accuracy"].metric.compute(),
                "pr": self.metrics["pr"].metric.compute(),
                "roc": self.metrics["roc"].metric.compute(),
                "weighted_auroc": self.metrics["weighted_auroc"].metric.compute(),
                "class_aurocs": self.metrics["class_aurocs"].metric.compute(),
                "conf_mat": self.metrics["conf_mat"].metric.compute(),
            }

        if self.prototype_metrics_cached:
            log.debug("returning cached metrics")
            result_dict = self.cached_results
            result_dict["accuracy"] = self.metrics["accuracy"].metric.compute()
            result_dict["pr"] = self.metrics["pr"].metric.compute()
            result_dict["roc"] = self.metrics["roc"].metric.compute()
            result_dict["weighted_auroc"] = self.metrics[
                "weighted_auroc"
            ].metric.compute()
            result_dict["class_aurocs"] = self.metrics["class_aurocs"].metric.compute()
            result_dict["conf_mat"] = self.metrics["conf_mat"].metric.compute()

        else:
            log.debug("calculating new metrics")
            result_dict = {
                metric_name: metric.metric.compute()
                for metric_name, metric in self.metrics.items()
            }

            result_dict["prototype_score"] = (
                min(result_dict["prototype_sparsity"], 1.0)
                + result_dict["prototype_stability"]
                + result_dict["prototype_consistency"]
            ) / 3

            self.prototype_metrics_cached = True
            self.cached_results = result_dict.copy()

        result_dict["acc_proto_score"] = (
            result_dict["prototype_score"] * result_dict["accuracy"]
        )

        return result_dict
