import json
import logging

import click
import torch

from . import datasets
from .activations import lookup_activation
from .backbones import construct_backbone, default_batch_sizes_for_backbone
from .cli import common_training_options
from .embedding import AddonLayers
from .losses import ClassAwareExtraCalculations
from .model_losses import loss_for_coefficients
from .models.deformable_protopnet import DeformableProtoPNet, DeformableTrainingSchedule
from .pixel_space import BilinearUpsampleActivations
from .train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from .train.scheduling import Phase, RSampleInitPhase
from .train.scheduling.early_stopping import ProjectPatienceEarlyStopping
from .train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


@click.command("train-deformable")
@common_training_options
def run(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    pre_project_phase_len=5,
    num_warm_pre_offset_epochs=5,
    phase_multiplier=1,  # for online augmentation
    latent_dim_multiplier_exp=None,
    joint_lr_step_size=5,
    post_project_phases=20,
    joint_epochs_per_phase=5,
    last_only_epochs_per_phase=10,
    cluster_coef=-0.8,
    separation_coef=0.08,
    l1_coef=0.01,
    num_addon_layers=0,
    fa_func="serial",
    fa_coef=None,
    num_prototypes_per_class=10,
    offset_weight_l2=0.8,
    orthogonality_loss=0.01,
    offset_bias_l2=0.8,
    cross_entropy=1,
    lr_step_gamma=0.2,
    lr_multiplier=1,
    prototype_dimension=(3, 3),
    class_specific=True,
    dry_run=False,
    verify=False,
    run_id=None,
    dataset="CUB200",
    proto_channels=None,
    save_on_proto_updates=False,
    rsample_init=False,
    activation_function="cosine",
):
    """
    Train a Deformable ProtoPNet.

    Args:
    - backbone: str - See backbones.py
    - pre_project_phase_len: int - number of epochs in each pre-project phase (warm-up, joint). Total preproject epochs is 2*pre_project_phase_len*phase_multiplier.
    - phase_multiplier: int - for each phase, multiply the number of epochs in that phase by this number
    - latent_dim_exp: int - expotential of 2 for the latent dimension of the prototype layer
    - joint_lr_step_size: int - number of epochs between each step in the joint learning rate scheduler. Multiplied by phase_multiplier.
    - last_only_epochs_per_phase: int - coefficient for clustering loss
    - post_project_phases: int - number of times to iterate between last-only, joint, project after the initial pre-project phases
    - cluster_coef: float - coefficient for clustering loss term
    - separation_coef: float - coefficient for separation loss term
    - l1_coef: float - coefficient for clustering loss
    - fa_func: str - one of "serial", "l2", or "square" to indicate which type of fine annotation loss to use. if None, fine annotation is deactivated.
    - fa_coef: float - coefficient for fine annotation loss term
    - num_prototypes_per_class: int - number of prototypes to create for each class
    - lr_multiplier: float - multiplier for learning rates. The base values are from protopnet's training.
    - class_specific: boolean - whether to bind prototypes to individual classes or allow them to be distributed unevenly based on training
    - dry_run: bool - Configure the training run, but do not execute it
    - preflight: bool - Configure a training run for a single epoch of all phases
    - verify: bool - Configure a training run for a single epoch of all phases
    - interpretable_metrics: bool - Whether to calculate interpretable metrics
    - save_on_proto_updates: bool - whether or not to save intermediate models to run pacmap on
    - rsample_init: bool - whether to initialize the prototypes are parts of the input images
    """

    if verify:
        log.info("Running in verification mode to check training will run all phases")
        pre_project_phase_len = 1
        post_project_phases = 0
        num_warm_pre_offset_epochs = 1
        joint_epochs_per_phase = 1
        last_only_epochs_per_phase = 1
        phase_multiplier = 1
        num_prototypes_per_class = 2
        latent_dim_multiplier_exp = -2
        proto_channels = None
        num_addon_layers = 1
        save_on_proto_updates = True
        prototype_dimension = (2, 2)

    log.info("Training config %s", json.dumps(locals()))
    log.log(
        logging.INFO if dry_run else logging.DEBUG,
        f'Running vanilla training as a {"dry_run" if dry_run else "full run"}',
    )

    setup = {
        "coefs": {
            "cluster": cluster_coef,
            "offset_weight_l2": offset_weight_l2,
            "separation": separation_coef,
            "orthogonality_loss": orthogonality_loss,
            "offset_bias_l2": offset_bias_l2,
            "l1": l1_coef,
            "fa": fa_coef,
            "cross_entropy": cross_entropy,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    post_forward_calculations = [ClassAwareExtraCalculations()]

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    if fa_coef is not None:
        post_forward_calculations.append(
            BilinearUpsampleActivations(split_dataloaders.image_size)
        )

    if type(prototype_dimension) is int:
        prototype_dimension = (prototype_dimension, prototype_dimension)

    activation = lookup_activation(activation_function)

    backbone = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=num_prototypes_per_class * split_dataloaders.num_classes,
        input_channels=backbone.latent_dimension[0],
        proto_channel_multiplier=(
            2**latent_dim_multiplier_exp
            if latent_dim_multiplier_exp is not None
            else None
        ),
        proto_channels=proto_channels,
        num_addon_layers=num_addon_layers,
    )

    vppn = DeformableProtoPNet(
        num_classes=split_dataloaders.num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
        backbone=backbone,
        add_on_layers=add_on_layers,
        activation=activation,
    )

    train_logger = WeightsAndBiasesTrainLogger(
        device=setup["device"],
        calculate_best_for=["accuracy"],
        wandb_plots=(
            ["pr", "roc", "conf_mat"] if split_dataloaders.num_classes <= 10 else None
        ),
    )

    protopnet_loss = loss_for_coefficients(
        setup["coefs"],
        fa_func=fa_func,
        class_specific_cluster=class_specific,
    ).to(setup["device"])

    # TODO: add L1 term only for last_only
    training_schedule = DeformableTrainingSchedule(
        model=vppn,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=protopnet_loss,
        last_only_loss=protopnet_loss,
        num_accumulation_batches=num_accumulation_batches,
        num_warm_epochs=pre_project_phase_len * phase_multiplier,
        num_preproject_joint_epochs=pre_project_phase_len * phase_multiplier,
        num_warm_pre_offset_epochs=num_warm_pre_offset_epochs * phase_multiplier,
        backbone_lr=1e-4 * lr_multiplier,
        add_on_lr=0.003 * lr_multiplier,
        prototype_lr=0.003 * lr_multiplier,
        pred_head_lr=1e-4 * lr_multiplier,
        conv_offset_lr=1e-4 * lr_multiplier,
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

    if rsample_init:
        training_schedule.phases.insert(
            0,
            Phase(
                train=RSampleInitPhase(dataloader=split_dataloaders.project_loader),
                duration=1,
                eval=training_schedule.default_eval_phase,
            ),
        )

    checkpoint_phases = {"project"} if save_on_proto_updates else set()

    ppn_trainer = MultiPhaseProtoPNetTrainer(
        device=setup["device"],
        metric_logger=train_logger,
        checkpoint_phase_starts=checkpoint_phases,
        checkpoint_phase_ends=checkpoint_phases,
        num_classes=split_dataloaders.num_classes,
    )

    if dry_run:
        log.info("Skipping training due to dry run: %s", training_schedule)
    else:
        ppn_trainer.train(
            run_id=run_id,
            model=vppn,
            training_schedule=training_schedule,
            target_metric_name="accuracy",
            early_stopping=ProjectPatienceEarlyStopping(project_patience=2),
        )
