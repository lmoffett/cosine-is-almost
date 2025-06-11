from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from protopnet.train.checkpointing import BestOnlyInMemoryModelCheckpointer
from protopnet.train.scheduling.types import Phase
from protopnet.train.trainer import MultiPhaseProtoPNetTrainer


class MockTrainLogger:
    def __init__(self):
        self.runs = {}
        self.current_run = None
        self.bests = {
            k: {
                "accuracy": {"prototypes_embedded": float("-inf"), "any": float("-inf")}
            }
            for k in ["train", "eval"]
        }
        self.metrics = []
        self.device = "cpu"

    def start_run(self, run_id):
        self.current_run = run_id
        self.runs[run_id] = []

    def end_epoch(
        self,
        metrics_dict,
        is_train,
        epoch_index,
        prototype_embedded_epoch,
        precalculated_metrics=None,
    ):
        if precalculated_metrics:
            self.metrics.append(
                {
                    "run_id": self.current_run,
                    "epoch": epoch_index,
                    "is_train": is_train,
                    "prototypes_embedded": prototype_embedded_epoch,
                    **precalculated_metrics,
                }
            )
            # Update bests for target metrics
            if not is_train:
                for metric_name, value in precalculated_metrics.items():
                    if value > self.bests["eval"][metric_name]["prototypes_embedded"]:
                        self.bests["eval"][metric_name]["prototypes_embedded"] = value


def test_trainer_minimal_init():
    """Test MultiPhaseProtoPNetTrainer initialization with minimal parameters"""
    metric_logger = MockTrainLogger()
    trainer = MultiPhaseProtoPNetTrainer(device="cpu", metric_logger=metric_logger)

    assert trainer.device == "cpu"
    assert trainer.save_threshold == 0.0
    assert trainer.compute_metrics_for_embed_only is True
    assert trainer.training_metrics is None


def test_trainer_full_init():
    """Test MultiPhaseProtoPNetTrainer initialization with all optional parameters"""
    metric_logger = MockTrainLogger()
    checkpoint_factory = Mock(
        return_value=BestOnlyInMemoryModelCheckpointer("test_run", "accuracy")
    )

    trainer = MultiPhaseProtoPNetTrainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        metric_logger=metric_logger,
        checkpoint_factory=checkpoint_factory,
        save_threshold=0.5,
        checkpoint_phase_starts={"warmup", "train"},
        checkpoint_phase_ends={"warmup", "train"},
        compute_metrics_for_embed_only=False,
        num_classes=5,
        log_list_metrics=["class_aurocs"],
    )

    assert trainer.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert trainer.save_threshold == 0.5
    assert trainer.compute_metrics_for_embed_only is False
    assert trainer.metric_logger == metric_logger
    assert trainer.checkpoint_factory == checkpoint_factory
    assert trainer._checkpoint_phase_starts == {"warmup", "train"}
    assert trainer._checkpoint_phase_ends == {"warmup", "train"}
    assert "class_aurocs" in trainer.log_list_metrics
    assert trainer.num_classes == 5


@pytest.fixture
def mock_model():
    model = Mock()
    model.to = Mock(return_value=model)
    model.prototypes_embedded = Mock(return_value=True)
    return model


@pytest.fixture
def mock_metric_logger():
    logger = Mock()
    logger.bests = {
        k: {"accuracy": {"prototypes_embedded": float(0.875), "any": float(0.89)}}
        for k in ["train", "eval"]
    }
    return logger


@pytest.fixture
def mock_checkpointer_factory():
    factory = Mock()
    factory.return_value = Mock()  # The factory creates a checkpointer
    return factory


@pytest.fixture
def trainer(mock_metric_logger, mock_checkpointer_factory):
    return MultiPhaseProtoPNetTrainer(
        device="cpu",
        metric_logger=mock_metric_logger,
        checkpoint_factory=mock_checkpointer_factory,
        save_threshold=0.7,
    )


class MockPhaseType:
    def __init__(self, name, step_metrics=None):
        self.name = name
        self._step_metrics = step_metrics or {"accuracy": 0.85, "loss": 0.1}

    def run_step(self, model, step_context):
        return self._step_metrics

    @contextmanager
    def phase_settings(self, model):
        yield


def test_basic_training_flow(trainer, mock_model):
    """Test the basic training flow with a single phase"""
    # Setup training schedule
    mock_train_phase = MockPhaseType("train_phase")
    mock_eval_phase = MockPhaseType("eval_phase")
    phase = Phase(train=mock_train_phase, eval=mock_eval_phase, duration=2)
    schedule = [phase]

    # Run training
    result = trainer.train(
        model=mock_model, training_schedule=schedule, target_metric_name="accuracy"
    )

    # Verify model was moved to device
    mock_model.to.assert_called_once_with("cpu")

    # Verify metric logging
    assert trainer.metric_logger.end_epoch.call_count == 4  # 2 steps × (train + eval)

    # Verify model was returned
    assert result == mock_model


def test_early_stopping_step(trainer, mock_model):
    """Test early stopping during a training step"""
    mock_early_stopping = Mock()
    mock_early_stopping.step_end_should_stop.return_value = (True, "low accuracy")
    mock_early_stopping.phase_end_should_stop.return_value = (False, None)

    mock_train_phase = MockPhaseType("train_phase")
    mock_eval_phase = MockPhaseType("eval_phase")
    phase = Phase(train=mock_train_phase, eval=mock_eval_phase, duration=5)
    schedule = [phase]

    result = trainer.train(
        model=mock_model,
        training_schedule=schedule,
        target_metric_name="accuracy",
        early_stopping=mock_early_stopping,
    )

    # Should stop after first step
    assert trainer.metric_logger.end_epoch.call_count == 2  # 1 step × (train + eval)
    assert result == mock_model


def test_early_stopping_phase(trainer, mock_model):
    """Test early stopping after completing phases"""
    mock_early_stopping = Mock()
    # Don't stop during steps
    mock_early_stopping.step_end_should_stop.return_value = (False, None)
    # Stop after 2 phases
    mock_early_stopping.phase_end_should_stop.side_effect = [
        (False, None),  # First phase completes
        (False, None),  # Second phase completes
        (True, "no improvement for 2 phases"),  # Third phase triggers stop
    ]

    mock_train_phase = MockPhaseType("train_phase")
    mock_eval_phase = MockPhaseType("eval_phase")

    schedule = [
        Phase(train=mock_train_phase, eval=mock_eval_phase, duration=2),
        Phase(train=mock_train_phase, eval=mock_eval_phase, duration=1),
        Phase(train=mock_train_phase, eval=mock_eval_phase, duration=3),
        Phase(
            train=MockPhaseType("never_runs"),
            eval=MockPhaseType("never_runs"),
            duration=100,
        ),
    ]

    result = trainer.train(
        model=mock_model,
        training_schedule=schedule,
        target_metric_name="accuracy",
        early_stopping=mock_early_stopping,
    )

    assert (
        trainer.metric_logger.end_epoch.call_count == 12
    )  # (2 + 1 + 3 steps) × (train + eval)
    assert result == mock_model


def test_checkpoint_saving(trainer, mock_model):
    """Test checkpoint saving logic"""
    mock_warm_phase = MockPhaseType("warm_phase", {"accuracy": 0.7})
    mock_eval_phase = MockPhaseType("eval_phase", {"accuracy": 0.9})  # Better
    mock_joint_phase = MockPhaseType("joint_phase", {"accuracy": 0.95})  # Better
    mock_project_phase = MockPhaseType(
        "project_phase", {"accuracy": 0.75}
    )  # Worse than .8
    schedule = [
        Phase(train=mock_warm_phase, eval=mock_eval_phase, duration=1),
        Phase(train=mock_joint_phase, eval=mock_joint_phase, duration=1),
        Phase(train=mock_project_phase, eval=mock_project_phase, duration=1),
    ]

    trainer._checkpoint_phase_starts = {"warm_phase"}
    trainer._checkpoint_phase_ends = {"project_phase"}

    result = trainer.train(
        model=mock_model,
        training_schedule=schedule,
        target_metric_name="accuracy",
        run_id="test_run",
    )

    checkpointer = trainer.checkpoint_factory.return_value

    # Check initial checkpoint
    checkpointer.save_checkpoint.assert_any_call(
        model=mock_model, step_index=0, metric=np.nan, phase=None, descriptor="initial"
    )

    # Check phase start checkpoint
    checkpointer.save_checkpoint.assert_any_call(
        model=mock_model,
        step_index=0,
        metric=None,
        phase="warm_phase",
        descriptor="phase_start",
    )

    # Check phase start checkpoint
    checkpointer.save_checkpoint.assert_any_call(
        model=mock_model,
        step_index=2,
        metric=0.75,
        phase="project_phase",
        descriptor="phase_end",
    )

    # Check best checkpoint saving
    checkpointer.save_best.assert_any_call(
        model=mock_model,
        step_index=0,
        metric=0.90,
        phase="warm_phase",
    )

    checkpointer.save_best.assert_any_call(
        model=mock_model,
        step_index=1,
        metric=0.95,
        phase="joint_phase",
    )

    # project wasn't called
    assert len(checkpointer.save_best.mock_calls) == 2, str(
        checkpointer.save_best.mock_calls
    )


def test_multiple_phases(trainer, mock_model):
    """Test training with multiple phases"""
    mock_phase1 = MockPhaseType("phase1")
    mock_phase2 = MockPhaseType("phase2")

    schedule = [
        Phase(train=mock_phase1, eval=mock_phase1, duration=2),
        Phase(train=mock_phase2, eval=mock_phase2, duration=1),
    ]

    result = trainer.train(
        model=mock_model, training_schedule=schedule, target_metric_name="accuracy"
    )

    # Verify total number of steps
    assert (
        trainer.metric_logger.end_epoch.call_count == 6
    )  # (2 + 1) steps × (train + eval)


def test_run_id_generation(trainer, mock_model):
    """Test that run_id is generated if not provided"""
    mock_phase = MockPhaseType("phase")
    schedule = [Phase(train=mock_phase, eval=mock_phase, duration=1)]

    with patch("random.Random") as mock_random:
        result = trainer.train(
            model=mock_model, training_schedule=schedule, target_metric_name="accuracy"
        )

    # Verify that random was used to generate run_id
    mock_random.assert_called_once()


def test_before_after_training_hooks(trainer, mock_model):
    """Test that before_training and after_training hooks are called if present"""

    class PhaseWithHooks(MockPhaseType):
        def before_training(self, **kwargs):
            self.before_called = True

        def after_training(self, **kwargs):
            self.after_called = True

    phase_type = PhaseWithHooks("phase_with_hooks")
    schedule = [Phase(train=phase_type, eval=MockPhaseType("eval"), duration=1)]

    result = trainer.train(
        model=mock_model, training_schedule=schedule, target_metric_name="accuracy"
    )

    assert hasattr(phase_type, "before_called")
    assert hasattr(phase_type, "after_called")
