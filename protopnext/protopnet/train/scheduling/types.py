from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, Union

import torch

from ...prototypical_part_model import ProtoPNet
from ..checkpointing import ModelCheckpointer
from ..logging.types import TrainLogger


@dataclass(frozen=True)
class PrePhaseSummary:
    """
    Describes where the phase is positioned in the entire training process.
    """

    name: str
    first_step: int
    initial_target_metric: Tuple[str, Union[int, float, torch.Tensor]]
    expected_last_step: int

    def complete(
        self,
        last_step: int,
        final_target_metric: Tuple[str, Union[int, float, torch.Tensor]],
    ) -> "PostPhaseSummary":
        return PostPhaseSummary(
            name=self.name,
            first_step=self.first_step,
            initial_target_metric=self.initial_target_metric,
            expected_last_step=self.expected_last_step,
            last_step=last_step,
            final_target_metric=final_target_metric,
        )


@dataclass(frozen=True)
class PostPhaseSummary(PrePhaseSummary):
    last_step: int
    final_target_metric: Tuple[str, Union[int, float, torch.Tensor]]


class SetTrainingLayers(Protocol):
    """Protocol for managing which layers should be trainable."""

    def set_training_layers(self, model: ProtoPNet, phase: "PhaseType") -> None:
        """
        Set which layers in the model should be trainable.

        Args:
            model: The model whose layers will be configured
            phase: The PhaseType this is being run against, which may contain information about how to set the layers
        """
        ...


@dataclass(frozen=True)
class StepContext:
    global_step: int
    step_in_phase: int


class PostForwardCalculation(Protocol):
    """Protocol for post-forward pass calculations."""

    def __call__(
        self, model_output: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process the model output and batch data after a forward pass.

        Args:
            model_output: The output from the model's forward pass
            batch: The input batch data

        Returns:
            Dictionary containing processed results
        """
        ...


class PhaseType(Protocol):
    """Protocol defining the interface for a training phase."""

    @property
    def name(self) -> str:
        """The name of this phase."""
        ...

    def before_training(
        self,
        model: ProtoPNet,
        metric_logger: TrainLogger,
        checkpointer: ModelCheckpointer,
        pre_phase_summary: PrePhaseSummary,
    ) -> Optional[Dict[str, Any]]:
        """
        Initialize the phase.

        Args:
            model: The model being trained

        Returns:
            Optional dictionary of initialization results
        """
        ...

    def after_training(
        self,
        model: ProtoPNet,
        metric_logger: TrainLogger,
        checkpointer: ModelCheckpointer,
        post_phase_summary: PostPhaseSummary,
    ) -> Optional[Dict[str, Any]]:
        """
        Clean up at the end of the phase.

        Args:
            model: The model being trained

        Returns:
            Optional dictionary of cleanup results
        """
        ...

    @contextmanager
    def phase_settings(self, model: ProtoPNet) -> Iterator[None]:
        """
        Context manager for setting up the phase.

        Args:
            model: The model being trained
        """
        ...

    def run_step(self, model: Any) -> Optional[Dict[str, Any]]:
        """
        Run one step of this phase.

        Args:
            model: The model being trained

        Returns:
            Optional dictionary of step results
        """
        ...


class IterativePhaseProtocol(Protocol):
    """Protocol for phases that repeat a sequence of sub-phases."""

    @property
    def phases(self) -> list:
        """List of phases to iterate through."""
        ...

    @property
    def iterations(self) -> int:
        """Number of times to repeat the phase sequence."""
        ...


@dataclass(frozen=True)
class Phase:
    train: PhaseType
    duration: int = 1
    eval: Optional[PhaseType] = None

    @property
    def name(self) -> str:
        return self.train.name

    def __repr__(self):
        maybe_eval_str = f"eval={self.eval.name}, " if self.eval else ""
        return f"Phase({self.name}, {maybe_eval_str}duration={self.duration})"


@dataclass(frozen=True)
class IterativePhase:
    phases: List[Phase]
    iterations: int

    def __repr__(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        phase_indent = "  " * (indent_level + 1)

        # Format each phase with proper indentation
        phase_str = ",\n".join([f"{phase_indent}{phase}" for phase in self.phases])

        return (
            f"{indent}IterativePhase(\n"
            f"{phase_str},\n"
            f"{phase_indent}iterations={self.iterations}\n"
            f"{indent})"
        )


class TrainingSchedule:
    def __init__(
        self,
        phases: List[Union[Phase, IterativePhase]],
        default_eval_phase: Phase,
    ):
        self._phases = phases
        self._default_eval_phase = default_eval_phase

    @property
    def phases(self) -> List[Union[Phase, IterativePhase]]:
        return self._phases

    @property
    def default_eval_phase(self) -> Phase:
        return self._default_eval_phase

    def apply_defaults(self, phase: Phase) -> Phase:
        if phase.eval is None:
            return Phase(phase.train, phase.duration, self.default_eval_phase)
        return phase

    def __iter__(self) -> Iterator[Phase]:
        """Iterator over all phases, including expanded iterative phases."""
        for item in self.phases:
            if isinstance(item, Phase):
                yield self.apply_defaults(item)
            elif isinstance(item, IterativePhase):
                for _ in range(item.iterations):
                    for phase in item.phases:
                        yield self.apply_defaults(phase)

    def __phase_str(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        parts = []

        for phase in self.phases:
            if isinstance(phase, IterativePhase):
                # Pass the current indent level to IterativePhase's repr
                phase_for_print = IterativePhase(
                    [self.apply_defaults(p) for p in phase.phases], phase.iterations
                )
                parts.append(phase_for_print.__repr__(indent_level))
            else:
                phase_for_print = self.apply_defaults(phase)
                parts.append(f"{indent}{phase_for_print}")

        return "\n".join(parts)

    def __repr__(self):
        return f"TrainingSchedule(\n{self.__phase_str()}\n)"


class EarlyStopping(Protocol):
    """
    Protocol for early stopping criteria.

    This object is stateful and should be called at the end of each step and phase to work correctly.
    """

    def step_end_should_stop(
        self,
        step: int,
        current_step_target_metric: float,
        phase_summary: PrePhaseSummary,
    ) -> bool:
        """
        Determine if training should stop after a step.
        """
        ...

    def phase_end_should_stop(self, phase_summary: PostPhaseSummary) -> bool:
        """
        Determine if training should stop after a phase.
        """
        ...
