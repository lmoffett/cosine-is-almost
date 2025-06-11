import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import wandb

from ...train.scheduling.types import EarlyStopping, PostPhaseSummary, PrePhaseSummary

log = logging.getLogger(__name__)


class ProjectPatienceEarlyStopping:
    """
    Concrete implementation of early stopping criteria for prototypical networks.
    Implements both project phase patience and post-project threshold checking.
    """

    def __init__(
        self,
        project_patience: int,
        post_project_threshold: float = 0.0,
        project_phase_name: str = "project",
    ):
        self.project_phase_name = project_phase_name
        self.project_patience = project_patience
        self.post_project_threshold = post_project_threshold

        self.best_preproject_metric = float("-inf")
        self.best_project_metric = float("-inf")
        self.last_eval_metric = float("-inf")
        self.phases_without_improvement = 0

    def step_end_should_stop(
        self,
        step: int,
        current_step_target_metric: float,
        phase_summary: PrePhaseSummary,
    ) -> Tuple[bool, str]:
        return (False, "ProjectPatienceEarlyStopping does not stop after steps.")

    def phase_end_should_stop(
        self, phase_summary: PostPhaseSummary
    ) -> Tuple[bool, str]:
        """
        Check if we should stop training after a phase.
        Always returns False if the phase is not the project phase.

        Args:
            phase_summary: The summary of the phase that just ended.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether to stop training
            and a string containing the reason for stopping.
        """
        if phase_summary.name != self.project_phase_name:
            return False, "Not the project phase."

        final_metric = (
            phase_summary.final_target_metric[1]
            if phase_summary.final_target_metric[1] is not None
            else float("-inf")
        )
        initial_metric = (
            phase_summary.initial_target_metric[1]
            if phase_summary.initial_target_metric[1] is not None
            else float("-inf")
        )

        if (
            initial_metric <= self.best_preproject_metric
            and final_metric <= self.best_project_metric
        ):
            self.phases_without_improvement += 1
        else:
            self.phases_without_improvement = 0

        # Update best metrics
        self.best_project_metric = max(self.best_project_metric, final_metric)
        self.best_preproject_metric = max(self.best_preproject_metric, initial_metric)

        if self.phases_without_improvement >= self.project_patience:
            return True, "Project phase patience exceeded."

        if final_metric <= self.post_project_threshold:
            return True, "Post-project threshold not exceeded."

        return False, "No stopping criteria met."


class SweepProjectEarlyStopping:
    """
    Percentile-based early stopping criteria that compares performance
    against all runs in a W&B sweep.
    """

    def __init__(
        self,
        percentile_threshold: float = 30.0,
        project_phase_name: str = "project",
        metric_name: str = "target_metric",
        min_runs_required: int = 10,
        brackets: Set[int] = [1, 3, 7, 15],
    ):
        """

        Args:
            percentile_threshold: The percentile threshold to compare against at each project. If the run is below this threshold, it terminates.
            project_phase_name: The name of the project phase to check for stopping.
            metric_name: The name of the metric to compare against.
            min_runs_required: The minimum number of runs needed for a valid comparison.
        """
        # Validate percentile threshold
        if not 0 <= percentile_threshold <= 100:
            raise ValueError("Percentile threshold must be between 0 and 100")

        # Validate minimum runs
        if min_runs_required <= 0:
            raise ValueError("Minimum required runs must be positive")

        if len(brackets) == 0 or any([b < 0 for b in brackets]):
            raise ValueError("Brackets must be non-empty and contain positive integers")

        self.percentile_threshold = percentile_threshold
        self.project_phase_name = project_phase_name
        self.metric_name = metric_name
        self.min_runs_required = min_runs_required
        self.current_project_count = 0
        self.project_metrics = []
        self.brackets = set(brackets)

    def _fetch_sweep_metrics(self, current_project: int) -> List[float]:
        """
        Fetch metrics from all runs in the sweep at the current project iteration.
        Gets the sweep from the current run's context.

        The metrics are logged with format 'project/{metric_name}' and are only logged
        during project evaluations. Each run may have a different number of project
        evaluations, so we need to properly order them.
        """
        metrics: List[float] = []
        current_run = wandb.run

        if not current_run or not current_run.sweep_id:
            raise ValueError("No active sweep found in current run")

        api = wandb.Api()
        sweep = api.sweep(
            f"{current_run.entity}/{current_run.project}/{current_run.sweep_id}"
        )

        metric_name = f"project/{self.metric_name}"

        for run in sweep.runs:
            # Get all logged project metrics for this run
            history = run.scan_history(keys=[metric_name])
            project_metrics = []

            for row in history:
                value = row.get(metric_name)
                if value is not None:
                    project_metrics.append(value)

            # If this run has reached the current project iteration, add its metric
            if len(project_metrics) > current_project:
                metrics.append(project_metrics[current_project])

        return metrics

    def _compute_percentile(
        self, current_project: int, metric: float
    ) -> Tuple[float, int]:
        """
        Compute the percentile of the current metric value and number of comparison runs.
        """

        # cache once we have enough runs to know the distribution
        if len(self.project_metrics) < 30:
            log.debug("Updating cache of project metrics")
            self.project_metrics = self._fetch_sweep_metrics(current_project)
            log.debug("Project metrics are %s", np.array(self.project_metrics))

        return (
            100
            * (
                np.sum(np.array(self.project_metrics) <= metric)
                / len(self.project_metrics)
            ),
            len(self.project_metrics),
        )

    def step_end_should_stop(
        self,
        step: int,
        current_step_target_metric: float,
        phase_summary: PrePhaseSummary,
    ) -> Tuple[bool, str]:
        return False, "Only checking at project boundaries"

    def phase_end_should_stop(
        self, phase_summary: PostPhaseSummary
    ) -> Tuple[bool, str]:
        if phase_summary.name != self.project_phase_name:
            return False, "Not the project phase."

        try:
            _, final_metric = phase_summary.final_target_metric

            if self.current_project_count not in self.brackets:
                message = f"Skipping percentile check at project at non-bracket project {self.current_project_count}"
                log.debug(message)
                return False, message

            # Check percentile at phase end
            final_percentile, num_runs = self._compute_percentile(
                self.current_project_count, final_metric
            )

            # Check if we have enough runs for comparison
            if num_runs < self.min_runs_required:
                message = (
                    f"Not enough runs for comparison (have {num_runs}, "
                    f"need {self.min_runs_required} at project {self.current_project_count})"
                )
                log.debug(message)
                return False, message

            if final_percentile <= self.percentile_threshold:
                return True, (
                    f"Performance in bottom {self.percentile_threshold}% "
                    f"(compared against {num_runs} runs at project {self.current_project_count})"
                )

            message = (
                f"Continuing training (current performance at {final_percentile:.1f} percentile "
                f"among {num_runs} runs at project {self.current_project_count})"
            )
            log.debug(message)
            return False, message

        finally:
            # Update current project count
            self.current_project_count += 1

    def get_current_percentile(self, metric: float) -> Tuple[float, int]:
        """
        Get the current performance percentile and number of runs.
        """
        return self._compute_percentile(self.current_project_count, metric)


class CombinedEarlyStopping:
    """
    Flexible wrapper class that combines multiple early stopping criteria.
    Returns True if any stopping criterion is met.
    """

    def __init__(
        self,
        stoppers: Dict[str, EarlyStopping],
    ):
        """
        Initialize with a dictionary of named stoppers.

        Args:
            stoppers: Dictionary mapping names to stopper instances
                        e.g. {"patience": patience_stopper, "hyperband": hyperband_stopper}
        """
        if not stoppers or len(stoppers) == 0 or not isinstance(stoppers, dict):
            raise ValueError("At least one early stopping criterion must be provided")

        self.stoppers = stoppers

    def step_end_should_stop(
        self,
        step: int,
        current_step_target_metric: float,
        phase_summary: PrePhaseSummary,
    ) -> Tuple[bool, str]:
        """Check all stoppers at step end."""
        reasons = []
        for name, stopper in self.stoppers.items():
            should_stop, reason = stopper.step_end_should_stop(
                step, current_step_target_metric, phase_summary
            )
            if should_stop:
                return True, f"{name}: {reason}"
            reasons.append(f"{name}: {reason}")

        return False, " AND ".join(reasons)

    def phase_end_should_stop(
        self, phase_summary: PostPhaseSummary
    ) -> Tuple[bool, str]:
        """Check all stoppers at phase end."""
        reasons = []
        for name, stopper in self.stoppers.items():
            should_stop, reason = stopper.phase_end_should_stop(phase_summary)
            if should_stop:
                return True, f"{name}: {reason}"
            reasons.append(f"{name}: {reason}")

        return False, " AND ".join(reasons)
