import contextlib
import logging
import random
import time
from typing import Collection, Optional, Union

import numpy as np
import torch

from ..prototypical_part_model import ProtoPNet
from ..utilities.naming import generate_random_phrase, k_letters
from ..utilities.trainer_utilities import is_single_valued_metric
from .checkpointing import FilesystemModelCheckpointerFactory, ModelCheckpointerFactory
from .logging.types import TrainLogger
from .metrics import TrainingMetrics
from .scheduling import PrePhaseSummary, StepContext
from .scheduling.types import EarlyStopping, TrainingSchedule

log = logging.getLogger(__name__)


def format_metrics(metrics):
    return "|".join(
        f"{k}:{v:.3g}" for k, v in metrics.items() if type(v) is float or type(v) is int
    )


class MultiPhaseProtoPNetTrainer:
    def __init__(
        self,
        device: Union[str, torch.device],
        metric_logger: TrainLogger,
        checkpoint_factory: Optional[ModelCheckpointerFactory] = None,
        training_metrics: Optional[TrainingMetrics] = None,
        save_threshold: float = 0.0,
        checkpoint_phase_starts: Collection = set(),
        checkpoint_phase_ends: Collection = set(),
        compute_metrics_for_embed_only=True,
        num_classes=None,
        log_list_metrics=None,
    ):
        """
        Trains a ProtoPNet model in multiple phases.

        Takes the TrainingSchedule object as input and executes each phase according to the provided parameters.
        There is no alternative trainer method, so this functionality represents the default behavior.
        """
        # Basic Configuration
        self._device = device

        # Metrics Configuration
        self.num_classes = num_classes
        self.log_list_metrics = log_list_metrics
        # Include class_aurocs and conf_mat by default if <= 10 classes
        if self.num_classes is not None and self.num_classes <= 10:
            if self.log_list_metrics is None:
                self.log_list_metrics = []
            self.log_list_metrics.append("class_aurocs")
            self.log_list_metrics.append("conf_mat")

        self.compute_metrics_for_embed_only = compute_metrics_for_embed_only
        self._metric_logger = metric_logger

        # DEPRECATED: will be removed in rewrite of hte metrics
        self.training_metrics = training_metrics

        # FIXME - the logger shouldn't need a device
        self._metric_logger.device = self.device

        # Checkpointing Configuration
        self.save_threshold = save_threshold

        if checkpoint_factory is None:
            self._checkpoint_factory = FilesystemModelCheckpointerFactory()
        else:
            self._checkpoint_factory = checkpoint_factory

        self._checkpoint_phase_starts = checkpoint_phase_starts
        self._checkpoint_phase_ends = checkpoint_phase_ends

    @property
    def device(self):
        return self._device

    @property
    def checkpoint_factory(self):
        return self._checkpoint_factory

    @property
    def metric_logger(self):
        return self._metric_logger

    def train(
        self,
        model: ProtoPNet,
        training_schedule: TrainingSchedule,
        target_metric_name: str,
        early_stopping: Optional[EarlyStopping] = None,
        run_id: Optional[str] = None,
    ):
        """
        Apply this trainer to a ProtoPNet model using the provided training schedule.

        Args:
            run_id: A unique identifier for this training run. If not provided, a random phrase will be generated.
            model: The ProtoPNet model to train
            training_schedule: The schedule of training phases to execute
            target_metric_name: The name of the target metric to use for early stopping
        """

        if run_id is None:
            name_rng = random.Random(int(time.time()))
            run_id = (
                f"{generate_random_phrase(name_rng)}-{k_letters(k=5, rng=name_rng)}"
            )

        checkpointer = self.checkpoint_factory(
            run_id=run_id, target_metric_name=target_metric_name
        )
        self.metric_logger.start_run(run_id)
        model = model.to(self.device)

        log.info("Training with the following schedule:")
        log.info("%s", repr(training_schedule))

        target_metric = None

        if (
            len(self._checkpoint_phase_ends) > 0
            or len(self._checkpoint_phase_starts) > 0
        ):
            checkpointer.save_checkpoint(
                model=model,
                step_index=0,
                metric=np.nan,
                phase=None,
                descriptor="initial",
            )

        phase_start = 0
        step_in_phase = -1

        context_args = {
            "model": model,
            "checkpointer": checkpointer,
            "metric_logger": self.metric_logger,
        }

        for phase in training_schedule:
            phase_start = phase_start + step_in_phase + 1
            pre_phase_summary = PrePhaseSummary(
                name=phase.name,
                first_step=phase_start,
                initial_target_metric=(target_metric_name, target_metric),
                expected_last_step=phase_start + phase.duration - 1,
            )

            if phase.name in self._checkpoint_phase_starts:
                checkpointer.save_checkpoint(
                    model=model,
                    step_index=pre_phase_summary.first_step,
                    metric=target_metric,
                    phase=phase.name,
                    descriptor="phase_start",
                )

            if hasattr(phase.train, "before_training"):
                phase.train.before_training(
                    pre_phase_summary=pre_phase_summary, **context_args
                )

            log.info(
                "starting phase %s to run for %s steps (%s-%s).",
                phase.name,
                phase.duration,
                pre_phase_summary.first_step,
                pre_phase_summary.expected_last_step,
            )

            for step_in_phase in range(phase.duration):
                step_context = StepContext(phase_start + step_in_phase, step_in_phase)

                def run_this_step(phase_type, is_eval):
                    train_eval_str = "eval" if is_eval else "train"
                    log.info(
                        "starting step %s %s[%s] (%s/%s in phase)",
                        step_context.global_step,
                        phase.name,
                        train_eval_str,
                        step_in_phase + 1,
                        phase.duration,
                    )
                    start = time.time()

                    with (
                        phase_type.phase_settings(model)
                        if hasattr(phase_type, "phase_settings")
                        else contextlib.nullcontext()
                    ):
                        maybe_metrics = phase_type.run_step(
                            model=model, step_context=step_context
                        )

                    end = time.time()
                    log.info(
                        f"step %s %s[%s] completed in {end - start:.3f} seconds",
                        step_context.global_step,
                        phase.name,
                        train_eval_str,
                    )

                    if maybe_metrics is not None:
                        maybe_metrics.update({"time": end - start})
                        metrics = maybe_metrics
                    else:
                        metrics = {"time": end - start}

                    metrics_for_log = metrics.copy()
                    if self.log_list_metrics:
                        for metric_name, value in metrics.items():
                            if not is_single_valued_metric(value):
                                if metric_name not in self.log_list_metrics:
                                    del metrics_for_log[metric_name]
                                else:
                                    metrics_for_log[metric_name] = list(value)

                            # log epoch metrics dict
                    log.info(
                        "step %s %s metrics: %s",
                        train_eval_str,
                        step_context.global_step,
                        format_metrics(metrics_for_log),
                    )

                    return metrics

                train_metrics = run_this_step(phase.train, is_eval=False)
                self.metric_logger.end_epoch(
                    {},
                    is_train=True,
                    epoch_index=step_context.global_step,
                    prototype_embedded_epoch=model.prototypes_embedded(),
                    precalculated_metrics=train_metrics,
                )

                eval_metrics = run_this_step(phase.eval, is_eval=True)
                target_metric = eval_metrics.get(target_metric_name, None)

                previous_best = self.metric_logger.bests["eval"][target_metric_name][
                    "prototypes_embedded"
                ]

                self.metric_logger.end_epoch(
                    {},
                    is_train=False,
                    epoch_index=step_context.global_step,
                    prototype_embedded_epoch=model.prototypes_embedded(),
                    precalculated_metrics=eval_metrics,
                )

                if model.prototypes_embedded():
                    if (
                        target_metric > previous_best
                        and target_metric > self.save_threshold
                    ):
                        model_path = checkpointer.save_best(
                            model=model,
                            step_index=step_context.global_step,
                            metric=target_metric,
                            phase=phase.name,
                        )
                        self.metric_logger.log_best_model(model_path)

                    else:
                        log.debug(
                            "skipping saving model state with %s %s",
                            target_metric_name,
                            target_metric,
                        )

                if early_stopping:
                    should_stop, reason = early_stopping.step_end_should_stop(
                        step_context.global_step, target_metric, pre_phase_summary
                    )
                    if should_stop:
                        log.info(
                            "Early stopping after %s epochs because %s",
                            step_context.global_step + 1,
                            reason,
                        )
                        return model

            post_phase_summary = pre_phase_summary.complete(
                last_step=step_context.global_step,
                final_target_metric=(target_metric_name, target_metric),
            )

            if hasattr(phase.train, "after_training"):
                phase.train.after_training(
                    post_phase_summary=post_phase_summary, **context_args
                )

            if phase.name in self._checkpoint_phase_ends:
                checkpointer.save_checkpoint(
                    model=model,
                    step_index=post_phase_summary.last_step,
                    metric=target_metric,
                    phase=phase.name,
                    descriptor="phase_end",
                )

            # parsimonious early stopping
            if early_stopping:
                should_stop, reason = early_stopping.phase_end_should_stop(
                    post_phase_summary
                )
                if should_stop:
                    log.info(
                        "Early stopping after %s epochs because %s",
                        step_context.global_step + 1,
                        reason,
                    )
                    return model

            # end of phase

        final_best_path = checkpointer.archive_best()
        self.metric_logger.log_best_model(final_best_path)

        log.info("Training complete after %s epochs", step_context.global_step + 1)
        return model


class ProtoTreeTrainer(MultiPhaseProtoPNetTrainer):
    pass
