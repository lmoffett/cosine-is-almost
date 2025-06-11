import unittest

import torch

from protopnet.activations import CosPrototypeActivation, L2Activation
from protopnet.backbones import construct_backbone
from protopnet.datasets.torch_extensions import TensorDatasetDict
from protopnet.embedding import AddonLayers
from protopnet.models.vanilla_protopnet import (
    ProtoPNetTrainingSchedule,
    VanillaProtoPNet,
)
from protopnet.prototype_layers import (
    ClassAwarePrototypeLayer,
    PrototypeFromSampleMeta,
    PrototypeWithMeta,
)


def metadatafy_tensors(tensors, num_classes):
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


def test_vanilla_protopnet():
    vppn = VanillaProtoPNet(
        backbone=construct_backbone("resnet18"),
        add_on_layers=AddonLayers(3 * 2),
        activation=L2Activation(),
        num_classes=3,
        num_prototypes_per_class=2,
    )

    input = torch.randn(10, 3, 224, 224)
    logits = vppn.forward(input)["logits"]

    assert logits.shape == (10, 3)


class TestProtopNetTraining(unittest.TestCase):
    def setUp(self):
        self.num_classes = 2
        self.num_prototypes_per_class = 2

        self.coefs = {
            "clst": -0.8,
            "offset_weight_l2": 0.8,
            "sep": 0.8,
            "orthogonality_loss": 0.01,
            "offset_bias_l2": 0.8,
            "l1": 0.01,
            "crs_ent": 2,
        }
        self.cos_activation = CosPrototypeActivation()
        self.l2_activation = L2Activation()

        self.cos_vppn = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(
                num_prototypes=self.num_classes * self.num_prototypes_per_class
            ),
            activation=self.cos_activation,
            num_classes=self.num_classes,
            num_prototypes_per_class=self.num_prototypes_per_class,
        )

        self.l2_vppn = VanillaProtoPNet(
            backbone=construct_backbone("resnet18"),
            add_on_layers=AddonLayers(
                num_prototypes=self.num_classes * self.num_prototypes_per_class
            ),
            activation=self.l2_activation,
            num_classes=self.num_classes,
            num_prototypes_per_class=self.num_prototypes_per_class,
        )

        data = torch.randn(2, 3, 224, 224)
        labels = torch.tensor([1, 0], dtype=torch.long)
        dataset = TensorDatasetDict(data, labels)
        self.deterministic_data = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=2
        )

    def test_cos_prune(self):
        input_data = torch.randn(2, 3, 100, 100)
        new_protos = self.cos_vppn.prototype_tensors.data
        new_protos[0] = new_protos[1]
        self.cos_vppn.prototype_layer.set_prototypes(
            metadatafy_tensors(new_protos, self.num_classes)
        )

        original_res = self.cos_vppn(input_data)

        self.cos_vppn.prune_duplicate_prototypes()
        post_prune_res = self.cos_vppn(input_data)

        assert torch.allclose(
            original_res["logits"], post_prune_res["logits"]
        ), post_prune_res

        assert len(self.cos_vppn.prototype_layer.prototype_tensors) == 3

    def test_l2_prune(self):
        input_data = torch.randn(2, 3, 100, 100)
        new_protos = self.l2_vppn.prototype_tensors
        new_protos[0] = new_protos[1]
        self.l2_vppn.prototype_layer.set_prototypes(
            metadatafy_tensors(new_protos, self.num_classes)
        )

        original_res = self.l2_vppn(input_data)

        self.l2_vppn.prune_duplicate_prototypes()
        post_prune_res = self.l2_vppn(input_data)

        assert torch.allclose(
            original_res["logits"], post_prune_res["logits"]
        ), post_prune_res

        assert len(self.l2_vppn.prototype_layer.prototype_tensors) == 3


def test_prototype_density_metrics():
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = ClassAwarePrototypeLayer(
        activation_function=cosine_activation,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=(2, 2),
    )

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 14, 14))

    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors, num_classes))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors, num_classes))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors, num_classes))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'


def test_protopnet_training_schedule_minimal_config(
    mock_protopnet, mock_dataloaders, mock_loss
):
    """Test ProtoPNetTrainingSchedule instantiation and iteration with minimal configuration"""
    schedule = ProtoPNetTrainingSchedule(
        model=mock_protopnet,
        train_loader=mock_dataloaders["train"],
        val_loader=mock_dataloaders["val"],
        project_loader=mock_dataloaders["project"],
        loss=mock_loss,
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


def test_protopnet_training_schedule_full_config(
    mock_protopnet, mock_dataloaders, mock_loss
):
    """Test ProtoPNetTrainingSchedule with all configuration options set"""
    schedule = ProtoPNetTrainingSchedule(
        model=mock_protopnet,
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

    # Check warm phase optimizer
    warm_phase = phase_dict["warm"]
    assert len(warm_phase.train.optimizer.param_groups) == 2
    assert warm_phase.train.optimizer.param_groups[0]["lr"] == 0.01  # add_on_lr

    # Check joint phase optimizer and scheduler
    joint_phase = phase_dict["joint"]
    assert len(joint_phase.train.optimizer.param_groups) == 3
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
