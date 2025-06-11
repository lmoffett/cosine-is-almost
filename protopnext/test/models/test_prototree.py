from unittest.mock import Mock

import numpy as np
import pytest

from protopnet.models.prototree import (
    ProtoTreeBackpropPhaseWithLeafUpdate,
    ProtoTreeTrainingSchedule,
    TrainProtoTreeLayersUsingProtoPNetNames,
)
from protopnet.train.scheduling.scheduling import ProjectPhase, PrunePrototypesPhase


def test_prototree_training_schedule_minimal(
    mock_prototree, mock_dataloaders, mock_loss
):
    """Test ProtoTreeTrainingSchedule with minimal/default configuration."""
    schedule = ProtoTreeTrainingSchedule(
        model=mock_prototree,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        loss=mock_loss,
        phase_config_kwargs={"device": "cpu"},
    )

    # Verify the schedule structure
    assert len(schedule.phases) == 4  # warm, joint, prune, project phases
    assert schedule.default_eval_phase is not None

    # Check phase durations
    phases = list(schedule)
    assert phases[0].duration == 30  # default num_warm_epochs
    assert phases[1].duration == 70  # default num_joint_epochs

    # Verify phase types
    assert type(phases[0].train) is ProtoTreeBackpropPhaseWithLeafUpdate
    assert phases[0].train.name == "warm"
    assert phases[0].train.set_training_layers_fn.train_backbone is False
    assert type(phases[1].train) is ProtoTreeBackpropPhaseWithLeafUpdate
    assert phases[1].train.set_training_layers_fn.train_backbone is True
    assert phases[1].train.name == "joint"
    assert type(phases[2].train) is PrunePrototypesPhase
    assert type(phases[3].train) is ProjectPhase


def test_prototree_training_schedule_maximal(
    mock_prototree, mock_dataloaders, mock_loss
):
    """Test ProtoTreeTrainingSchedule with all parameters set."""
    backbone_override_layers = ["0.weight", "0.bias"]

    schedule = ProtoTreeTrainingSchedule(
        model=mock_prototree,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        loss=mock_loss,
        num_warm_epochs=15,
        num_joint_epochs=35,
        joint_epochs_before_lr_milestones=15,
        num_milestones=3,
        backbone_lr=2e-5,
        prototype_lr=0.002,
        add_on_lr=0.002,
        backbone_layers_override_lr=(backbone_override_layers, 3e-5),
        lr_weight_decay=0.001,
        lr_step_gamma=0.3,
        adamw_weight_decay=0.001,
        adamw_eps=1e-08,
        phase_config_kwargs={"device": "cpu"},
        num_accumulation_batches=2,
    )

    # Verify the schedule structure
    assert len(schedule.phases) == 4
    assert schedule.default_eval_phase is not None

    # Check phase durations
    phases = list(schedule)
    assert phases[0].duration == 15  # custom num_warm_epochs
    assert phases[1].duration == 35  # custom num_joint_epochs

    # Verify phase types
    assert type(phases[0].train) is ProtoTreeBackpropPhaseWithLeafUpdate
    assert type(phases[1].train) is ProtoTreeBackpropPhaseWithLeafUpdate
    assert type(phases[2].train) is PrunePrototypesPhase
    assert type(phases[3].train) is ProjectPhase

    # Verify learning rate milestones
    lr_milestones = list(np.linspace(30, 50, num=3, dtype=int))
    joint_phase = phases[1].train
    assert joint_phase.scheduler.milestones == {m: 1 for m in lr_milestones}
    assert joint_phase.scheduler.gamma == 0.3


def test_default_backbone_overrides():
    """Test that backbone_overrides defaults to None"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=True,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
    )
    assert trainer.backbone_overrides is None


@pytest.mark.parametrize("initial_grad", [True, False])
def test_overwrite_grad_with_no_overrides(initial_grad):
    """Test that overwrite_grad_epoch_settings doesn't modify params when no overrides are specified"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
    )

    mock_param = Mock()
    mock_param.requires_grad = initial_grad

    trainer.overwrite_grad_epoch_settings(
        "backbone.layer1", mock_param, "train_backbone", False
    )
    assert mock_param.requires_grad == initial_grad


def test_overwrite_grad_with_matching_override():
    """Test that overwrite_grad_epoch_settings modifies params when override matches"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
        backbone_overrides=["layer4"],
    )

    mock_param = Mock()
    mock_param.requires_grad = False

    trainer.overwrite_grad_epoch_settings(
        "backbone.layer4", mock_param, "train_backbone", False
    )
    assert mock_param.requires_grad


def test_overwrite_grad_with_non_matching_override():
    """Test that overwrite_grad_epoch_settings doesn't modify params when override doesn't match"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
        backbone_overrides=["layer4"],
    )

    mock_param = Mock()
    mock_param.requires_grad = False

    trainer.overwrite_grad_epoch_settings(
        "backbone.layer3", mock_param, "train_backbone", False
    )
    assert not mock_param.requires_grad


def test_overwrite_grad_with_multiple_overrides():
    """Test that overwrite_grad_epoch_settings works with multiple overrides"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
        backbone_overrides=["layer3", "layer4"],
    )

    mock_param1 = Mock()
    mock_param1.requires_grad = False
    mock_param2 = Mock()
    mock_param2.requires_grad = False

    trainer.overwrite_grad_epoch_settings(
        "backbone.layer3", mock_param1, "train_backbone", False
    )
    trainer.overwrite_grad_epoch_settings(
        "backbone.layer4", mock_param2, "train_backbone", False
    )

    assert mock_param1.requires_grad
    assert mock_param2.requires_grad


def test_set_training_layers_integration():
    """Test the integration with set_training_layers method"""
    trainer = TrainProtoTreeLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=True,
        backbone_overrides=["layer4"],
    )

    # Create a mock model with named parameters
    mock_prototree = Mock()
    mock_params = {
        "backbone.layer3": Mock(requires_grad=True),
        "backbone.layer4": Mock(requires_grad=True),
        "add_on_layers.conv1": Mock(requires_grad=False),
    }
    mock_prototree.named_parameters = lambda: mock_params.items()

    # Mock the phase
    mock_phase = Mock()

    # Call set_training_layers
    trainer.set_training_layers(mock_prototree, mock_phase)

    # Check that the parameters were set correctly
    assert not mock_params[
        "backbone.layer3"
    ].requires_grad  # Should be False (no override)
    assert mock_params["backbone.layer4"].requires_grad  # Should be True (has override)
    assert mock_params[
        "add_on_layers.conv1"
    ].requires_grad  # Should be True (train_add_on_layers=True)
