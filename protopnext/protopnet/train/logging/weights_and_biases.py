import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import wandb

from ...utilities.trainer_utilities import is_single_valued_metric
from .types import TrainLogger

log = logging.getLogger(__name__)


class WeightsAndBiasesTrainLogger(TrainLogger):
    def __init__(
        self,
        device="cpu",
        calculate_best_for=["accu"],
        wandb_plots=None,
    ):
        super().__init__(
            calculate_best_for=calculate_best_for,
            device=device,
        )

        # Plots to display on wandb dashboard ["pr", "conf_mat", "roc"]
        self.wandb_plots = wandb_plots

    def start_run(
        self, run_id: Optional[str] = None, run_meta: Optional[Dict[str, Any]] = {}
    ) -> str:
        wandb.init(
            project=run_meta.get(
                "project", os.environ.get("PPNXT_WANDB_PROJECT", "test")
            ),
            id=run_id,
        )
        return wandb.run.id

    def log_metrics(
        self,
        is_train,
        prototypes_embedded_state=False,
        precalculated_metrics=None,
        step=None,
    ):
        if is_train is True:  # noqa E712 - allowing non-boolean values
            metric_group, metrics, commit = "train", self.train_metrics, False
        elif is_train is False:  # noqa E712 - allowing non-boolean values
            metric_group, metrics, commit = "eval", self.val_metrics, True
        else:
            metric_group, metrics, commit = is_train, {}, False

        metrics_for_log = {
            f"{metric_group}/{name}": metric.compute()
            for name, metric in metrics.items()
            if metric._update_called
        }

        if precalculated_metrics:
            for key, value in precalculated_metrics.items():
                if not is_single_valued_metric(value):
                    if self.wandb_plots and key in self.wandb_plots:
                        # Code adjusted from wandb plot
                        if key == "pr":
                            precision, recall, _ = value
                            self.wandb_pr(precision, recall, step, metric_group, 10000)
                        elif key == "roc":
                            fpr, tpr, _ = value
                            self.wandb_roc(fpr, tpr, step, metric_group, 10000)
                        elif key == "conf_mat":
                            conf_mat = value
                            self.wandb_conf_mat(conf_mat, step, metric_group)
                    else:
                        log.debug('Skipping metric logging of "%s" as it is not a single value.', key)
                else:
                    metrics_for_log[f"{metric_group}/{key}"] = value

        wandb.log(metrics_for_log, step=step, commit=commit)

        for metric in metrics.values():
            # TODO - it's very bad that we're resetting metrics in a logging function
            metric.reset()

    def process_new_best(self, metric_name, metric_value, step):
        """
        This method is called whenever a new "best" value of a metric is found with the value of the metric, the current, step,
        and whether the prototype layer is embedded or not.

        This updates the weights and biases run summary with the new best value of the metric and the step at which it was found.
        """
        wandb.run.summary[metric_name] = metric_value
        wandb.run.summary[f"{metric_name}_step"] = step

    def log_best_model(self, model_path: Union[str, Path]):
        wandb.run.summary["best_model"] = str(Path(model_path).absolute())

    def end_epoch(
        self,
        epoch_metrics_dict,
        is_train,
        epoch_index,
        prototype_embedded_epoch,
        precalculated_metrics=None,
    ):
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
            step=epoch_index,
            prototypes_embedded_state=prototype_embedded_epoch,
            precalculated_metrics=precalculated_metrics,
        )

    def wandb_pr(self, precision, recall, step, metric_group, max_points=10000):
        """
        Log a Precision-Recall curve plot to Weights & Biases (wandb).

        This function avoids using wandb's ``wandb.plot.pr_curve()``
        method as it cannot calculate the necessary values incrementally.
        Instead, this function processes already calculated Precision and
        Recall values and visualizes them to the wandb dashboard using
        ``wandb.plot_table``.

        Parameters
        ----------
        precision : list[torch.Tensor]
            Precision values at different classification thresholds
            for each class.
        recall : list[torch.Tensor]
            Recall values at different classification thresholds for
            each class.
        step : int
            Current step number to associate metrics with on wandb.
        metric_group : str
            Label used to differentiate between phases of the training
            schedule (``train``, ``eval``).

        Returns
        ----------
        None: The function logs the Precision-Recall curve plot directly to wandb.
        """
        data = []

        avail_points = max_points // len(precision)
        for i, (cur_precision, cur_recall) in enumerate(zip(precision, recall)):
            label = f"class_{i}"
            num_points = len(cur_precision)
            gap = max(1, num_points // avail_points + 1)
            for j in range(0, num_points, gap):
                data.append(
                    [
                        label,
                        round(float(cur_precision[j]), 3),
                        round(float(cur_recall[j]), 3),
                    ]
                )

        columns = ["class", "precision", "recall"]
        vega_spec_name = "wandb/area-under-curve/v0"
        fields = {
            "x": "recall",
            "y": "precision",
            "class": "class",
        }
        string_fields = {
            "title": f"Downsampled Precision-Recall Curve ({metric_group})"
        }

        plot = wandb.plot_table(
            data_table=wandb.Table(
                columns=columns,
                data=data,
            ),
            vega_spec_name=vega_spec_name,
            fields=fields,
            string_fields=string_fields,
            split_table=False,
        )

        wandb.log({f"{metric_group}/pr": plot}, step=step)

    def wandb_roc(self, fpr, tpr, step, metric_group, max_points=10000):
        """
        Log a ROC curve plot to Weights & Biases (wandb).

        This function avoids using wandb's ``wandb.plot.roc_curve()``
        method as it cannot calculate the roc values incrementally. Instead,
        this function processes already calculated FPR and TPR values and
        visualizes them to the wandb dashboard using ``wandb.plot_table``.

        Parameters
        ----------
        fpr : list[torch.Tensor]]
            False positive rates at different classification thresholds
            for each class.
        tpr : list[torch.Tensor]
            True positive rates at different classification thresholds
            for each class.
        step : int
            Current step number to associate metrics with on wandb.
        metric_group : str
            Label used to differentiate between phases of the training
            schedule (``train``, ``eval``).

        Returns
        ----------
        None: The function logs the ROC curve plot directly to wandb.
        """
        data = []

        avail_points = max_points // len(fpr)
        for i, (cur_fpr, cur_tpr) in enumerate(zip(fpr, tpr)):
            label = f"class_{i}"
            num_points = len(cur_fpr)
            gap = max(1, num_points // avail_points + 1)
            for j in range(0, num_points, gap):
                data.append(
                    [label, round(float(cur_fpr[j]), 3), round(float(cur_tpr[j]), 3)]
                )

        columns = ["class", "fpr", "tpr"]
        vega_spec_name = "wandb/area-under-curve/v0"
        fields = {
            "x": "fpr",
            "y": "tpr",
            "class": "class",
        }
        string_fields = {
            "title": f"Downsampled ROC Curve ({metric_group})",
            "x-axis-title": "False positive rate",
            "y-axis-title": "True positive rate",
        }

        plot = wandb.plot_table(
            data_table=wandb.Table(
                columns=columns,
                data=data,
            ),
            vega_spec_name=vega_spec_name,
            fields=fields,
            string_fields=string_fields,
            split_table=False,
        )

        wandb.log({f"{metric_group}/roc": plot}, step=step)

    def wandb_conf_mat(self, conf_mat, step, metric_group):
        """
        Log a confusion matrix plot to Weights & Biases (wandb).

        This function avoids using wandb's ``wandb.plot.confusion_matrix()``
        method as it cannot calculate the confusion matrix incrementally. Instead,
        this function processes an already calculated confusion matrix and
        visualizes the counts to the wandb dashboard using ``wandb.plot_table``.

        Parameters
        ----------
        conf_mat : torch.Tensor
            Confusion matrix representing the counts of predicted classes for each
            actual class.
        step : int
            Current step number to associate metrics with on wandb.
        metric_group : str
            Label used to differentiate between phases of the training schedule
            (``train``, ``eval``).

        Returns
        ----------
        None: The function logs the confusion matrix plot directly to wandb.
        """
        data = []
        for i, row in enumerate(conf_mat):
            for j, count in enumerate(row):
                data.append([f"class_{i}", f"class_{j}", int(count)])

        columns = ["Actual", "Predicted", "nPredictions"]
        vega_spec_name = "wandb/confusion_matrix/v1"
        fields = {
            "Actual": "Actual",
            "Predicted": "Predicted",
            "nPredictions": "nPredictions",
        }
        string_fields = {"title": f"Confusion Matrix Curve ({metric_group})"}

        plot = wandb.plot_table(
            data_table=wandb.Table(
                columns=columns,
                data=data,
            ),
            vega_spec_name=vega_spec_name,
            fields=fields,
            string_fields=string_fields,
            split_table=False,
        )

        wandb.log({f"{metric_group}/conf_mat": plot}, step=step)

    @staticmethod
    def log_backdrops(backdrop_dict, step=None):
        # log dict to wandb
        wandb.log(backdrop_dict, step=step)
