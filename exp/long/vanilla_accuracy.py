import json
import logging

import torch
from protopnet import datasets
from protopnet.activations import lookup_activation
from protopnet.backbones import construct_backbone, default_batch_sizes_for_backbone
from protopnet.embedding import AddonLayers
from protopnet.losses import ClassAwareExtraCalculations
from protopnet.model_losses import loss_for_coefficients
from protopnet.models.vanilla_protopnet import (
    ProtoPNetTrainingSchedule,
    VanillaProtoPNet,
)
from protopnet.train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from protopnet.train.scheduling.early_stopping import ProjectPatienceEarlyStopping
from protopnet.train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


def train(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    pre_project_phase_len: int = 10,
    phase_multiplier: int = 1,
    joint_lr_step_size: int = 5,
    post_project_phases: int = 20,
    cluster_coef: float = -0.8,
    separation_coef: float = 0.08,
    joint_epochs_per_phase: int = 10,
    last_only_epochs_per_phase: int = 20,
    num_prototypes_per_class: int = 10,
    lr_multiplier: float = 1.0,
    lr_step_gamma: float = 0.1,
    latent_dim_multiplier_exp: int = 0,
    dataset: str = "CUB200",
    activation_function: str = "cosine",
):

    log.info("Training config %s", json.dumps(locals()))

    setup = {
        "num_prototypes_per_class": num_prototypes_per_class,
        "coefs": {
            "cluster": cluster_coef,
            "separation": separation_coef,
            "l1": 1e-4,
            "cross_entropy": 1,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    post_forward_calculations = [ClassAwareExtraCalculations()]

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    activation = lookup_activation(activation_function)

    backbone = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=setup["num_prototypes_per_class"]
        * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=2**latent_dim_multiplier_exp,
    )

    vppn = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=split_dataloaders.num_classes,
        num_prototypes_per_class=setup["num_prototypes_per_class"],
    )

    train_logger = WeightsAndBiasesTrainLogger(
        device=setup["device"],
        calculate_best_for=["accuracy"],
    )

    protopnet_loss = loss_for_coefficients(
        setup["coefs"],
        class_specific_cluster=True,
    ).to(setup["device"])

    # TODO: add L1 term only for last_only
    training_schedule = ProtoPNetTrainingSchedule(
        model=vppn,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=protopnet_loss,
        last_only_loss=protopnet_loss,
        num_accumulation_batches=num_accumulation_batches,
        num_warm_epochs=pre_project_phase_len * phase_multiplier,
        num_preproject_joint_epochs=pre_project_phase_len * phase_multiplier,
        backbone_lr=1e-4 * lr_multiplier,
        add_on_lr=0.003 * lr_multiplier,
        prototype_lr=0.003 * lr_multiplier,
        pred_head_lr=1e-4 * lr_multiplier,
        weight_decay=1e-3,
        joint_lr_step_size=joint_lr_step_size * phase_multiplier,
        joint_lr_step_gamma=lr_step_gamma,
        last_only_epochs_per_project=last_only_epochs_per_phase * phase_multiplier,
        joint_epochs_per_phase=joint_epochs_per_phase * phase_multiplier,
        post_project_phases=post_project_phases,
        phase_config_kwargs={
            "device": setup["device"],
            "post_forward_calculations": post_forward_calculations,
        },
    )

    ppn_trainer = MultiPhaseProtoPNetTrainer(
        device=setup["device"],
        metric_logger=train_logger,
        num_classes=split_dataloaders.num_classes,
    )

    ppn_trainer.train(
        model=vppn,
        training_schedule=training_schedule,
        target_metric_name="accuracy",
        early_stopping=ProjectPatienceEarlyStopping(project_patience=2),
    )
