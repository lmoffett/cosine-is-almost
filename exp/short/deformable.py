import json
import logging

import numpy as np
import torch
from protopnet import datasets
from protopnet.activations import lookup_activation
from protopnet.backbones import construct_backbone, default_batch_sizes_for_backbone
from protopnet.embedding import AddonLayers
from protopnet.losses import ClassAwareExtraCalculations
from protopnet.model_losses import loss_for_coefficients
from protopnet.models.deformable_protopnet import (
    DeformableProtoPNet,
    DeformableTrainingSchedule,
)
from protopnet.train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from protopnet.train.scheduling.early_stopping import (
    CombinedEarlyStopping,
    ProjectPatienceEarlyStopping,
    SweepProjectEarlyStopping,
)
from protopnet.train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


def train(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    post_project_phases: int = 32,
    lr_multiplier: float = 1.0,
    num_warm_epochs_at_lr1: int = 5,
    joint_steps_per_phase_at_lr1: int = 10,
    last_only_steps_per_joint_step: float = 1.0,
    lr_step_per_joint_phase_2exp: int = 0,
    lr_step_gamma: float = 0.1,
    cluster_coef: float = -0.8,
    separation_coef: float = 0.08,
    l1_coef: float = 1e-4,
    orthogonality_loss_coef: float = 0.01,
    num_prototypes_per_class: int = 14,
    latent_dim_multiplier_exp: int = -2,
    dataset: str = "CUB200",
    activation_function: str = "cosine",
):

    log.info("Training config %s", json.dumps(locals()))
    if lr_multiplier == 0.0:
        lr_multiplier = 0.05
    elif lr_multiplier > 100:
        lr_multiplier = 100

    if joint_steps_per_phase_at_lr1 <= 0:
        joint_steps_per_phase_at_lr1 = 1

    setup = {
        "num_prototypes_per_class": num_prototypes_per_class,
        "coefs": {
            "cluster": cluster_coef,
            "separation": separation_coef,
            "l1": l1_coef,
            "cross_entropy": 1,
            "orthogonality_loss": orthogonality_loss_coef,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    post_forward_calculations = [ClassAwareExtraCalculations()]

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    if "resnet" in backbone:
        batch_sizes = {k: v - 10 for k, v in batch_sizes.items()}

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    activation = lookup_activation(activation_function)

    backbone = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=setup["num_prototypes_per_class"]
        * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=2**latent_dim_multiplier_exp,
    )

    vppn = DeformableProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        # deformable paper reports best results with 2x2 prototypes
        prototype_dimension=(2, 2),
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

    def scale_int_by_lr(x):
        """
        Scale an integer by the lr multiplier, and return at least 1.

        Since the learning rate multipliers are log distributed near zero, the rate modification is cbrt(lr_multiplier).
        """
        return max(1, int(x / np.cbrt(lr_multiplier)))

    joint_phase_steps = scale_int_by_lr(joint_steps_per_phase_at_lr1)
    joint_lr_step_size = max(
        1, int(joint_phase_steps * (2**lr_step_per_joint_phase_2exp))
    )

    schedule_args = dict(
        num_warm_epochs=scale_int_by_lr(num_warm_epochs_at_lr1),
        num_warm_pre_offset_epochs=scale_int_by_lr(num_warm_epochs_at_lr1),
        num_preproject_joint_epochs=joint_phase_steps,
        backbone_lr=1e-4 * lr_multiplier,
        add_on_lr=0.003 * lr_multiplier,
        prototype_lr=0.003 * lr_multiplier,
        pred_head_lr=1e-4 * lr_multiplier,
        conv_offset_lr=1e-4 * lr_multiplier,
        weight_decay=1e-3,
        joint_lr_step_size=joint_lr_step_size,
        joint_lr_step_gamma=lr_step_gamma,
        last_only_epochs_per_project=max(
            1, int(joint_phase_steps * last_only_steps_per_joint_step)
        ),
        joint_epochs_per_phase=joint_phase_steps,
        post_project_phases=post_project_phases,
    )

    log.info("Training schedule %s", json.dumps(schedule_args))

    training_schedule = DeformableTrainingSchedule(
        model=vppn,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=protopnet_loss,
        last_only_loss=protopnet_loss,
        num_accumulation_batches=num_accumulation_batches,
        **schedule_args,
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

    early_stopping = CombinedEarlyStopping(
        {
            "project_patience": ProjectPatienceEarlyStopping(project_patience=2),
            "global": SweepProjectEarlyStopping(
                metric_name="accuracy",
                min_runs_required=8,
                percentile_threshold=50.0,
                brackets=[1, 3, 7, 15],  # max 32 epochs
            ),
        }
    )

    ppn_trainer.train(
        model=vppn,
        training_schedule=training_schedule,
        target_metric_name="accuracy",
        early_stopping=early_stopping,
    )
