import json
import logging

import numpy as np
import torch
from protopnet import datasets
from protopnet.activations import lookup_activation
from protopnet.backbones import construct_backbone, default_batch_sizes_for_backbone
from protopnet.embedding import LightweightAddonLayers
from protopnet.losses import ClassAwareExtraCalculations
from protopnet.model_losses import loss_for_coefficients
from protopnet.models.st_protopnet import STProtoPNet, STProtoPNetTrainingSchedule
from protopnet.models.vanilla_protopnet import VanillaProtoPNet
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
    num_addon_layers: int = 2,
    joint_steps_per_phase_at_lr1: int = 10,
    last_only_steps_per_joint_step: float = 1.0,
    lr_step_per_joint_phase_2exp: int = 0,
    lr_step_gamma: float = 0.1,
    cluster_coef: float = -0.8,
    support_separation_coef: float = 0.48,
    trivial_separation_coef: float = 0.08,
    orthogonality_loss_coef: float = 0.001,
    closeness_loss_coef: float = 1.0,
    discrimination_loss_coef: float = 1.0,
    l1_coef: float = 1e-4,
    num_prototypes_per_class: int = 14,
    dataset: str = "CUB200",
    activation_function: str = "cosine",
    ortho_p_norm: int = 1,
    proto_channels: int = 64,
):

    log.info("Training config %s", json.dumps(locals()))
    if lr_multiplier == 0.0:
        lr_multiplier = 0.05
    elif lr_multiplier > 100:
        lr_multiplier = 100

    if joint_steps_per_phase_at_lr1 <= 0:
        joint_steps_per_phase_at_lr1 = 1

    if num_prototypes_per_class % 2 != 0:
        raise ValueError("Number of prototypes per class must be even for ST-ProtoPNet")

    shared_coefs = {
        "cluster": cluster_coef,
        "l1": l1_coef,
        "cross_entropy": 1,
        "orthogonality_loss": orthogonality_loss_coef,
    }

    setup = {
        "support_coefs": {
            **shared_coefs,
            "separation": support_separation_coef,
            "closeness_loss": closeness_loss_coef,
        },
        "trivial_coefs": {
            **shared_coefs,
            "separation": trivial_separation_coef,
            "discrimination_loss": discrimination_loss_coef,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    post_forward_calculations = [ClassAwareExtraCalculations()]

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    activation = lookup_activation(activation_function)

    upsample_add_on_layer = bool("vgg" not in backbone)

    if upsample_add_on_layer:
        backbone = construct_backbone(backbone)
    else:
        # this is the case for VGG19 where it is 14x14 by default
        backbone = construct_backbone(
            backbone,
            pretrained=True,
            final_maxpool=False,
            final_relu=True,
        )

    add_on_layers = LightweightAddonLayers(
        num_prototypes=num_prototypes_per_class * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channels=proto_channels,
        num_addon_layers=num_addon_layers,
        is_upsampled=upsample_add_on_layer,
    )

    support_vppn = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=split_dataloaders.num_classes,
        num_prototypes_per_class=num_prototypes_per_class // 2,
    )

    trivial_vppn = VanillaProtoPNet(
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=split_dataloaders.num_classes,
        num_prototypes_per_class=num_prototypes_per_class // 2,
    )

    st_protopnet = STProtoPNet(models=[support_vppn, trivial_vppn])

    train_logger = WeightsAndBiasesTrainLogger(
        device=setup["device"],
        calculate_best_for=["accuracy"],
    )

    # trainining loss terms
    support_protopnet_loss = loss_for_coefficients(
        setup["support_coefs"],
        class_specific_cluster=True,
        ortho_p_norm=ortho_p_norm,
    ).to(setup["device"])

    trivial_protopnet_loss = loss_for_coefficients(
        setup["trivial_coefs"],
        class_specific_cluster=True,
        ortho_p_norm=ortho_p_norm,
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
        num_preproject_joint_epochs=joint_phase_steps,
        backbone_lr=1e-4 * lr_multiplier,
        add_on_lr=0.003 * lr_multiplier,
        prototype_lr=0.003 * lr_multiplier,
        pred_head_lr=1e-4 * lr_multiplier,
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

    training_schedule = STProtoPNetTrainingSchedule(
        model=st_protopnet,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        num_accumulation_batches=num_accumulation_batches,
        support_protopnet_loss=support_protopnet_loss,
        trivial_protopnet_loss=trivial_protopnet_loss,
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
        model=st_protopnet,
        training_schedule=training_schedule,
        target_metric_name="accuracy",
        early_stopping=early_stopping,
    )
