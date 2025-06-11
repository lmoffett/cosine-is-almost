import copy
import unittest

import pytest
import torch

from protopnet.activations import CosPrototypeActivation, L2Activation
from protopnet.backbones import construct_backbone
from protopnet.datasets.torch_extensions import TensorDatasetDict
from protopnet.embedding import AddonLayers
from protopnet.models.st_protopnet import (
    CachedForward,
    STProtoPNet,
    STProtoPNetBackpropPhase,
    STProtoPNetClassificationPhase,
    STProtoPNetTrainingSchedule,
)
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
from protopnet.train.scheduling.scheduling import TrainLayersUsingProtoPNetNames


def test_cache_forward():
    def forward_fn(x: torch.Tensor):
        return x.norm(p=2)

    # Pick two inputs that have distinct outputs from one another.
    input_1 = torch.Tensor([1.0, 2.0, 3.0])
    input_2 = torch.Tensor([4.0, 5.0, 6.0])

    # Outputs before forward is decorated with cache functionality.
    output_1 = forward_fn(input_1)
    output_2 = forward_fn(input_2)

    # Cache default state is disabled and empty.
    cached_forward_fn = CachedForward(forward_fn)
    assert (
        cached_forward_fn.enable_cache == False
    ), f"Expected cache to be disabled initially. Got {cached_forward_fn.enable_cache}"
    assert (
        cached_forward_fn.cache is None
    ), f"Expected cache to be None initially. Got {cached_forward_fn.cache}"

    # Cache state: disabled, empty.
    assert torch.allclose(
        output_1, cached_forward_fn(input_1)
    ), f"Failed to match expected output_1 of {output_1}"
    assert torch.allclose(
        output_2, cached_forward_fn(input_2)
    ), f"Failed to match expected output_2 of {output_1}"
    assert (
        cached_forward_fn.cache is None
    ), f"Expected cache to be None after the forward pass with cache disabled. Got {cached_forward_fn.cache}"

    assert not torch.allclose(
        cached_forward_fn(input_1), cached_forward_fn(input_2)
    ), f"Expected different outputs for different inputs, but got the same values."

    # Cache state: enabled, empty.
    cached_forward_fn.toggle_cache(toggle=True)
    assert (
        cached_forward_fn.enable_cache == True
    ), f"Expected cache to be enabled after toggle. Got {cached_forward_fn.enable_cache}"
    assert (
        cached_forward_fn.cache is None
    ), f"Expected cache to be None after enabling cache. Got {cached_forward_fn.cache}"

    # Cache state: enabled, filled with output_1.
    cached_ouput_1 = cached_forward_fn(input_1)
    assert torch.allclose(
        output_1, cached_forward_fn.cache
    ), f"Expected cache to be {output_1}, but got {cached_forward_fn.cache}"

    # Because cache is enabled, this forward should be true regardless of input
    # The logic might seem counterintuitive, but it just means cache is working.
    assert torch.allclose(
        cached_ouput_1, cached_forward_fn(input_2)
    ), f"Cache should return the same result for different inputs when cache is enabled."

    # Cache state: enabled, empty.
    cached_forward_fn.reset_cache()
    assert (
        cached_forward_fn.cache is None
    ), f"Expected cache to be reset to None after calling reset_cache. Got {cached_forward_fn.cache}"

    # Cache state: enabled, filled with output_2.
    cached_ouput_2 = cached_forward_fn(input_2)
    assert torch.allclose(
        output_2, cached_forward_fn.cache
    ), f"Expected cache to be {output_2}, but got {cached_forward_fn.cache}"

    # Similar to before, we expect the input to the forward function to be ignored.
    # This test ensures that the cache is being used correctly.
    assert torch.allclose(
        cached_ouput_2, cached_forward_fn(input_1)
    ), f"Cache should return the same result regardless of input when cache is enabled. Expected {cached_ouput_2}, but got {cached_forward_fn(input_1)}"

    # Cache state: disabled, empty.
    cached_forward_fn.toggle_cache(toggle=False)
    assert (
        cached_forward_fn.enable_cache == False
    ), f"Expected cache to be disabled after toggle. Got {cached_forward_fn.enable_cache}"
    assert (
        cached_forward_fn.cache is None
    ), f"Expected cache to be None after disabling cache. Got {cached_forward_fn.cache}"

    # Cache state: enabled, empty.
    assert torch.allclose(
        output_1, cached_forward_fn(input_1)
    ), f"Expected cache to return {output_1} after toggling cache off and on, but got {cached_forward_fn(input_1)}"
    assert torch.allclose(
        output_2, cached_forward_fn(input_2)
    ), f"Expected cache to return {output_2} after toggling cache off and on, but got {cached_forward_fn(input_2)}"


def test_st_protopnet():
    # considerations, do we need to check the model forward cache behavior
    support_vppn = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(3 * 2),
        activation=L2Activation(),
        num_classes=3,
        num_prototypes_per_class=2,
    )
    trivial_vppn = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(3 * 2),
        activation=L2Activation(),
        num_classes=3,
        num_prototypes_per_class=2,
    )

    st_vppn = STProtoPNet(models=[support_vppn, trivial_vppn])

    input = torch.randn(10, 3, 224, 224)
    output = st_vppn.forward(input)

    support_logits = output[0]["logits"]
    trivial_logits = output[1]["logits"]

    assert support_logits.shape == (10, 3)
    assert trivial_logits.shape == (10, 3)


def test_stprotopnet_training_schedule_minimal_config(
    mock_stprotopnet, mock_dataloaders, mock_loss
):
    """Test ProtoPNetTrainingSchedule instantiation and iteration with minimal configuration"""
    schedule = STProtoPNetTrainingSchedule(
        model=mock_stprotopnet,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        support_protopnet_loss=mock_loss,
        trivial_protopnet_loss=mock_loss,
        phase_config_kwargs={"device": torch.device("cpu")},
    )

    # One iterative phase
    assert len(list(schedule.phases)) == 5

    # Check phase sequence
    phase_names = [phase.name for phase in schedule]
    expected_sequence = ["warm"] + ["joint", "project", "last_only"] * (12 + 1)
    assert phase_names == expected_sequence

    # Verify phase config is applied
    for phase in schedule:
        if hasattr(phase.train, "device"):
            assert phase.train.device == torch.device("cpu")


def test_stprotopnet_training_schedule_full_config(
    mock_stprotopnet, mock_dataloaders, mock_loss
):
    """Test ProtoPNetTrainingSchedule with all configuration options set"""
    schedule = STProtoPNetTrainingSchedule(
        model=mock_stprotopnet,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        support_protopnet_loss=mock_loss,
        trivial_protopnet_loss=mock_loss,
        num_accumulation_batches=2,
        add_on_lr=0.01,
        num_warm_epochs=5,
        num_preproject_joint_epochs=5,
        backbone_lr=1e-5,
        prototype_lr=0.01,
        pred_head_lr=1e-5,
        weight_decay=1e-4,
        joint_lr_step_size=3,
        joint_lr_step_gamma=0.1,
        last_only_epochs_per_project=10,
        joint_epochs_per_phase=5,
        post_project_phases=6,
        phase_config_kwargs={"device": torch.device("cpu")},
    )

    # Verify all optimizers are configured correctly
    phase_dict = {phase.name: phase for phase in schedule}

    # Check warm phase optimizer (recall that we are working with 2 sub-models)
    warm_phase = phase_dict["warm"]
    assert (
        len(warm_phase.train.optimizer.param_groups) == 3
    )  # (2 layers x 2 sub_models)
    assert warm_phase.train.optimizer.param_groups[0]["lr"] == 0.01  # add_on_lr

    # Check joint phase optimizer and scheduler (recall extra sub_models layers)
    joint_phase = phase_dict["joint"]
    assert len(joint_phase.train.optimizer.param_groups) == 4
    assert joint_phase.train.scheduler.step_size == 3
    assert joint_phase.train.scheduler.gamma == 0.1

    # Verify iterative phase structure
    num_total_phases = len(list(schedule))
    expected_phases = (
        1
        + 1  # warm
        + 1  # preproject joint
        + (6 * 3)  # initial project
        + 1  # post_project_phases * (last_only + joint + project)  # final last_only
    )
    assert num_total_phases == expected_phases


def test_stprotopnet_set_sub_model_training_layers(mock_protopnet):
    """
    This test ensures the parameters in the branches/sub models have their
    gradient caclculations correctly required.

    Note:
        - This test is also targeting the overriden `ProtoPNet.named_parameters(.)`
            method within ST-ProtoPNet.
    """
    st_vppn = STProtoPNet(models=[mock_protopnet, copy.deepcopy(mock_protopnet)])

    trainer = TrainLayersUsingProtoPNetNames(
        train_backbone=False,
        train_add_on_layers=True,
        train_prototype_layer=True,
        train_prototype_prediction_head=False,
    )

    trainer.set_training_layers(st_vppn, None)

    backbone = st_vppn.backbone

    # Check that none of the backbone parameters require gradients
    for param in backbone.parameters():
        assert (
            param.requires_grad is False
        ), f"Backbone parameter {param} requires grad!"

    for model_idx, sub_model in enumerate(st_vppn.models):

        # Check that none of the add_on_layers parameters require gradients
        for param in sub_model.add_on_layers.parameters():
            assert (
                param.requires_grad is True
            ), f"Add-on layer parameter {param} requires grad in model {model_idx}!"

        # Check that none of the prototype_layer parameters require gradients
        for param in sub_model.prototype_layer.parameters():
            assert (
                param.requires_grad is True
            ), f"Prototype layer parameter {param} requires grad in model {model_idx}!"

        # Check that none of the prototype_prediction_head parameters require gradients
        for (
            param
        ) in sub_model.prototype_prediction_head.class_connection_layer.parameters():
            assert (
                param.requires_grad is False
            ), f"Prototype prediction head parameter {param} does NOT requires grad in model {model_idx}!"


@pytest.mark.parametrize(
    "batch_indices, num_accumulation_batches, expected_model_selection",
    [
        ([0, 1, 2, 3, 4], 1, [0, 1, 0, 1, 0]),
        ([0, 1, 2, 3, 4], 2, [0, 0, 1, 1, 0]),
        ([0, 1, 2, 3, 4], 3, [0, 0, 0, 1, 1]),
    ],
)
def test_sub_model_index_selection(
    batch_indices, num_accumulation_batches, expected_model_selection
):
    # Calculate the model selection based on _select_model_idx method
    model_selection = [
        STProtoPNetClassificationPhase._select_model_idx(
            None, idx, num_accumulation_batches
        )
        for idx in batch_indices
    ]

    # Assert the model selection matches the expected result
    assert (
        model_selection == expected_model_selection
    ), f"Expected {expected_model_selection}, got {model_selection}"


def test_set_sub_model_gradient(mock_stprotopnet):
    mock_trivial_model = mock_stprotopnet.models[0]
    mock_support_model = mock_stprotopnet.models[1]

    def all_layer_has_required_grad_set_to(mock_sub_model, expected_grad: bool):
        layers = [
            mock_sub_model.prototype_layer,
            mock_sub_model.prototype_prediction_head,
        ]
        for layer in layers:
            for param in layer.parameters():
                if param.requires_grad != expected_grad:
                    return False
        return True

    STProtoPNetBackpropPhase._set_sub_model_required_gradient(
        None, mock_trivial_model, True
    )
    STProtoPNetBackpropPhase._set_sub_model_required_gradient(
        None, mock_support_model, False
    )

    # assert that the layers have required grad set to true for trivial and false for support
    assert all_layer_has_required_grad_set_to(
        mock_trivial_model, True
    ), "Trivial model layers should have required_grad set to True"
    assert all_layer_has_required_grad_set_to(
        mock_support_model, False
    ), "Support model layers should have required_grad set to False"

    STProtoPNetBackpropPhase._set_sub_model_required_gradient(
        None, mock_trivial_model, False
    )
    STProtoPNetBackpropPhase._set_sub_model_required_gradient(
        None, mock_support_model, True
    )

    # assert that the layers have required grad set to false for trivial and true for support
    assert all_layer_has_required_grad_set_to(
        mock_trivial_model, False
    ), "Trivial model layers should have required_grad set to False"
    assert all_layer_has_required_grad_set_to(
        mock_support_model, True
    ), "Support model layers should have required_grad set to True"
