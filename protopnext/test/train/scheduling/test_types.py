from contextlib import contextmanager
from typing import ContextManager, Iterator

import pytest

from protopnet.train.scheduling.types import IterativePhase, Phase, TrainingSchedule


# Mock implementations for testing
class SimplePhaseType:
    """Simple implementation of PhaseType protocol for testing"""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def before_training(self, model, metrics_logger, checkpointer, pre_phase_summary):
        return None

    def after_training(self, model, metrics_logger, checkpointer, post_phase_summary):
        return None

    @contextmanager
    def phase_settings(self, model) -> Iterator[None]:
        yield

    def run_step(self, model):
        return None


@pytest.fixture
def joint_phase():
    return SimplePhaseType("joint")


@pytest.fixture
def warm_phase():
    return SimplePhaseType("warm")


@pytest.fixture
def project_phase():
    return SimplePhaseType("project")


@pytest.fixture
def default_eval_phase(project_phase):
    return Phase(project_phase)


@pytest.fixture
def basic_phase(joint_phase):
    return Phase(joint_phase)


@pytest.fixture
def basic_schedule(basic_phase, default_eval_phase):
    return TrainingSchedule([basic_phase], default_eval_phase)


def test_phase_creation_with_minimal_args(joint_phase):
    """Test Phase creation with just the required train argument"""
    phase = Phase(joint_phase)
    assert phase.train.name == "joint"
    assert phase.duration == 1
    assert phase.eval is None
    assert phase.name == "joint"


def test_phase_creation_with_all_args(warm_phase, project_phase):
    """Test Phase creation with all possible arguments specified"""
    phase = Phase(train=warm_phase, duration=5, eval=project_phase)
    assert phase.train.name == "warm"
    assert phase.duration == 5
    assert phase.eval.name == "project"
    assert phase.name == "warm"


def test_phase_repr(joint_phase, project_phase):
    """Test the string representation of Phase objects"""
    # Test minimal phase
    phase1 = Phase(joint_phase)
    assert repr(phase1) == "Phase(joint, duration=1)"

    # Test phase with all arguments
    phase2 = Phase(joint_phase, duration=3, eval=project_phase)
    assert repr(phase2) == "Phase(joint, eval=project, duration=3)"


def test_iterative_phase_creation_and_repr(joint_phase, warm_phase):
    """Test creation and string representation of IterativePhase"""
    phases = [Phase(joint_phase), Phase(warm_phase, duration=2)]
    iterative_phase = IterativePhase(phases=phases, iterations=3)
    expected_repr = (
        "IterativePhase(\n"
        "  Phase(joint, duration=1),\n"
        "  Phase(warm, duration=2),\n"
        "  iterations=3\n"
        ")"
    )
    assert repr(iterative_phase) == expected_repr


def test_training_schedule_nested_iterative_phase_repr(
    joint_phase, warm_phase, project_phase
):
    """Test the string representation of nested iterative phases in a training schedule"""

    inner_phases = [Phase(warm_phase, duration=2), Phase(project_phase, duration=1)]
    iterative = IterativePhase(phases=inner_phases, iterations=2)

    # Create schedule with nested structure
    schedule = TrainingSchedule(
        phases=[Phase(joint_phase, duration=1), iterative],
        default_eval_phase=Phase(project_phase),
    )

    expected_repr = (
        "TrainingSchedule(\n"
        "  Phase(joint, eval=project, duration=1)\n"
        "  IterativePhase(\n"
        "    Phase(warm, eval=project, duration=2),\n"
        "    Phase(project, eval=project, duration=1),\n"
        "    iterations=2\n"
        "  )\n"
        ")"
    )

    assert repr(schedule) == expected_repr


def test_training_schedule_iteration_with_single_phase(basic_schedule):
    """Test iteration over a schedule with a single regular phase"""
    phases = list(basic_schedule)
    assert len(phases) == 1
    assert phases[0].train.name == "joint"
    assert phases[0].eval.name == "project"  # Default eval applied


def test_training_schedule_iteration_with_iterative_phase(
    joint_phase, warm_phase, default_eval_phase
):
    """Test iteration over a schedule containing an IterativePhase"""
    iterative_phase = IterativePhase(
        phases=[Phase(joint_phase), Phase(warm_phase)], iterations=2
    )
    schedule = TrainingSchedule([iterative_phase], default_eval_phase)

    phases = list(schedule)
    assert len(phases) == 4  # 2 phases * 2 iterations
    assert [p.train.name for p in phases] == ["joint", "warm", "joint", "warm"]
    assert all(p.eval.name == "project" for p in phases)  # Default eval applied to all


def test_training_schedule_with_mixed_phases(
    joint_phase, warm_phase, default_eval_phase
):
    """Test schedule with both regular and iterative phases"""
    regular_phase = Phase(joint_phase)
    iterative_phase = IterativePhase(phases=[Phase(warm_phase)], iterations=2)
    schedule = TrainingSchedule(
        phases=[regular_phase, iterative_phase], default_eval_phase=default_eval_phase
    )

    phases = list(schedule)
    assert len(phases) == 3  # 1 regular + (1 phase * 2 iterations)
    assert [p.train.name for p in phases] == ["joint", "warm", "warm"]


def test_training_schedule_respects_custom_eval_phase(
    joint_phase, warm_phase, default_eval_phase
):
    """Test that custom eval phases aren't overridden by default"""
    custom_eval = Phase(warm_phase)
    phase = Phase(joint_phase, eval=custom_eval)
    schedule = TrainingSchedule([phase], default_eval_phase)

    phases = list(schedule)
    assert len(phases) == 1
    assert phases[0].eval.name == "warm"  # Custom eval preserved
