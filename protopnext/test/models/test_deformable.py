import copy

import pytest
import torch

from protopnet.activations import ConvolutionalSharedOffsetPred, CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.datasets.torch_extensions import TensorToDictDatasetAdapter
from protopnet.embedding import AddonLayers
from protopnet.models.deformable_protopnet import (
    DeformableProtoPNet,
    DeformableTrainingSchedule,
)
from protopnet.prediction_heads import LinearClassPrototypePredictionHead
from protopnet.prototype_layers import (
    DeformablePrototypeLayer,
    PrototypeFromSampleMeta,
    PrototypeWithMeta,
)
from protopnet.prototypical_part_model import ProtoPNet


def test_deform_model_construction():
    backbone = construct_backbone("resnet18")
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))
    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
    }

    prototype_shape = (num_classes * 2, 512, 2, 2)

    offset_predictor = ConvolutionalSharedOffsetPred(
        prototype_shape=prototype_shape, input_feature_dim=prototype_shape[1]
    )

    proto_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        offset_predictor=offset_predictor,
        activation_function=proto_activation,
        prototype_dimension=(2, 2),
    )

    prediction_head = LinearClassPrototypePredictionHead(**prototype_config)

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


def test_deformable_protopnet_construction():
    backbone = construct_backbone("resnet18")

    activation = CosPrototypeActivation()

    num_classes = 3
    num_prototypes_per_class = 10

    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes_per_class * num_classes,
        proto_channel_multiplier=0,
        num_addon_layers=0,
    )

    protopnet = DeformableProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=3,
        num_prototypes_per_class=10,
        k_for_topk=1,
        prototype_dimension=(3, 3),
        prototype_dilation=1,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = protopnet.forward(input)["logits"]

    assert logits.shape == (10, 3)


def test_deform_max_self_similarity():
    x = torch.randn(2, 512, 7, 7)
    prototype_tensors = x[:, :, 2, 2].unsqueeze(-1).unsqueeze(-1)

    offset_predictor = ConvolutionalSharedOffsetPred(
        prototype_shape=prototype_tensors.shape,
        input_feature_dim=prototype_tensors.shape[1],
    )

    # Dummy offset predictor that spits out zeros of the correct shape
    offset_predictor_wrapper = lambda x: torch.zeros_like(offset_predictor(x))

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    proto_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        offset_predictor=offset_predictor_wrapper,
        activation_function=proto_activation,
        prototype_dimension=(1, 1),
    )
    prototype_layer.prototype_tensors = torch.nn.Parameter(prototype_tensors)

    observed_activations = prototype_layer(x)["prototype_activations"]

    assert observed_activations[0, 0, 2, 2] >= 0.99
    assert observed_activations[1, 1, 2, 2] >= 0.99


def test_deform_max_self_similarity_multipart():
    x = torch.randn(2, 512, 7, 7)
    prototype_tensors = x[:, :, :3, :3]

    offset_predictor = ConvolutionalSharedOffsetPred(
        prototype_shape=prototype_tensors.shape,
        input_feature_dim=prototype_tensors.shape[1],
    )

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    proto_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        offset_predictor=offset_predictor,
        activation_function=proto_activation,
        prototype_dimension=(3, 3),
    )
    prototype_layer.prototype_tensors = torch.nn.Parameter(prototype_tensors)

    observed_activations = prototype_layer(x)["prototype_activations"]

    assert observed_activations[0, 0, 1, 1] >= 0.99
    assert observed_activations[1, 1, 1, 1] >= 0.99


def test_deform_max_self_similarity_multipart_dilation_2():
    x = torch.randn(2, 512, 7, 7)
    prototype_tensors = torch.zeros_like(x[:, :, :2, :2])
    prototype_tensors[:, :, 0, 0] = x[:, :, 0, 0]
    prototype_tensors[:, :, 0, 1] = x[:, :, 0, 2]
    prototype_tensors[:, :, 1, 0] = x[:, :, 2, 0]
    prototype_tensors[:, :, 1, 1] = x[:, :, 2, 2]

    offset_predictor = ConvolutionalSharedOffsetPred(
        prototype_shape=prototype_tensors.shape,
        input_feature_dim=prototype_tensors.shape[1],
        prototype_dilation=2,
    )

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    proto_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        offset_predictor=offset_predictor,
        activation_function=proto_activation,
        prototype_dimension=(3, 3),
    )
    prototype_layer.prototype_tensors = torch.nn.Parameter(prototype_tensors)

    observed_activations = prototype_layer(x)["prototype_activations"]

    assert observed_activations[0, 0, 1, 1] >= 0.99
    assert observed_activations[1, 1, 1, 1] >= 0.99


@pytest.mark.skip("needs a new training loop")
def test_offset_predictor_learns():
    backbone = construct_backbone("resnet18")
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))
    prototype_config = {
        "prototype_class_identity": prototype_class_identity,
        "num_classes": num_classes,
    }

    prototype_shape = (num_classes * 2, 512, 2, 2)

    offset_predictor = ConvolutionalSharedOffsetPred(
        prototype_shape=prototype_shape, input_feature_dim=prototype_shape[1]
    )
    original_offset_weights = copy.deepcopy(offset_predictor.offset_predictor.weight)

    proto_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        offset_predictor=offset_predictor,
        activation_function=proto_activation,
        prototype_dimension=(2, 2),
    )

    prediction_head = LinearClassPrototypePredictionHead(**prototype_config)

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )

    device = "cpu"

    warm_pre_offset_optimizer_specs = [
        {
            "params": protopnet.parameters(),
            "lr": 0.00001,
            "weight_decay": 1e-3,
        },
    ]

    optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)

    data = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 0], dtype=torch.long)
    dataset = TensorToDictDatasetAdapter(torch.utils.data.TensorDataset(data, labels))
    deterministic_data = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2
    )

    # TODO: run training

    assert (
        torch.norm(offset_predictor.offset_predictor.weight - original_offset_weights)
        > 1e-4
    )


def test_prototype_density_metrics_deformable():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        activation_function=cosine_activation,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=(2, 2),
    )

    def metadatafy_tensors(tensors):
        return [
            PrototypeWithMeta(
                prototype_tensor=tensors[i],
                meta=PrototypeFromSampleMeta(
                    sample_id=b"test_id",
                    target=i % num_classes,
                    hash=b"hash",
                    latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                ),
            )
            for i in range(tensors.shape[0])
        ]

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 14, 14))

    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'


def test_deformable_training_schedule_minimal_config(
    mock_deformable, mock_dataloaders, mock_loss
):
    """Test DeformableTrainingSchedule instantiation and iteration with minimal configuration"""
    schedule = DeformableTrainingSchedule(
        model=mock_deformable,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        loss=mock_loss,
        phase_config_kwargs={"device": torch.device("cpu")},
    )

    # One iterative phase
    assert (
        len(list(schedule.phases)) == 6
    )  # warm, warm_pre_offset, joint, project, iterative, last_only

    # Check phase sequence
    phase_names = [phase.name for phase in schedule]
    expected_sequence = (
        ["warm", "warm_pre_offset", "joint", "project"]
        + ["last_only", "joint", "project"] * 12
        + ["last_only"]
    )
    assert phase_names == expected_sequence

    # Verify phase config is applied
    for phase in schedule:
        if hasattr(phase.train, "device"):
            assert phase.train.device == torch.device("cpu")


def test_deformable_training_schedule_full_config(
    mock_deformable, mock_dataloaders, mock_loss
):
    """Test DeformableTrainingSchedule with all configuration options set"""
    schedule = DeformableTrainingSchedule(
        model=mock_deformable,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        loss=mock_loss,
        last_only_loss=mock_loss,
        num_accumulation_batches=2,
        add_on_lr=0.01,
        num_warm_epochs=5,
        num_preproject_joint_epochs=5,
        backbone_lr=1e-5,
        prototype_lr=0.01,
        pred_head_lr=1e-5,
        conv_offset_lr=1e-4,
        weight_decay=1e-4,
        joint_lr_step_size=3,
        joint_lr_step_gamma=0.1,
        last_only_epochs_per_project=10,
        joint_epochs_per_phase=5,
        num_warm_pre_offset_epochs=8,
        post_project_phases=6,
        phase_config_kwargs={"device": torch.device("cpu")},
    )

    # Verify all optimizers are configured correctly
    phase_dict = {phase.name: phase for phase in schedule}

    # Check warm phase optimizer
    warm_phase = phase_dict["warm"]
    assert len(warm_phase.train.optimizer.param_groups) == 2
    assert warm_phase.train.optimizer.param_groups[0]["lr"] == 0.01  # add_on_lr

    # Check warm_pre_offset phase optimizer
    warm_pre_offset_phase = phase_dict["warm_pre_offset"]
    assert len(warm_pre_offset_phase.train.optimizer.param_groups) == 4

    # Check joint phase optimizer and scheduler
    joint_phase = phase_dict["joint"]
    assert (
        len(joint_phase.train.optimizer.param_groups) == 4
    )  # add_on, proto, backbone, offset_pred
    assert joint_phase.train.scheduler.step_size == 3
    assert joint_phase.train.scheduler.gamma == 0.1

    # Verify iterative phase structure
    num_total_phases = len(list(schedule))
    expected_phases = (
        1
        + 1  # warm
        + 1  # preproject joint
        + 1  # warm_pre_offset
        + (6 * 3)  # initial project
        + 1  # post_project_phases * (last_only + joint + project)  # final last_only
    )
    assert num_total_phases == expected_phases
