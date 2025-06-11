import json
import logging

import click
import torch

from . import datasets
from .activations import lookup_activation
from .backbones import construct_backbone
from .cli import common_training_options
from .embedding import AddonLayers
from .model_losses import loss_for_coefficients
from .models.prototree import ProtoTree, ProtoTreeTrainingSchedule
from .pixel_space import BilinearUpsampleActivations
from .train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from .train.scheduling.scheduling import RSampleInitPhase
from .train.scheduling.types import Phase
from .train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


@click.command("train-prototree")
@common_training_options
def run(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    warm_up_phase_len=30,
    joint_phase_len=70,
    phase_multiplier=1,  # for online augmentation
    latent_dim_multiplier_exp=None,
    phase_relative_lr_multiplier=1,
    num_milestones=5,
    num_addon_layers=1,
    activation_function="exp_l2",
    fa_func="serial",
    fa_coef=None,
    depth=9,
    pruning_threshold: float = 0.01,
    log_probabilities: bool = False,
    k_for_topk: int = 1,
    lr_step_gamma=0.5,
    dry_run=False,
    verify=False,
    save_on_proto_updates=False,
    dataset="cub200",
    proto_channels=None,
    rsample_init=False,
    weight_decay=0,
    run_id=None,
):
    """
    Train a ProtoTree.

    Args:
        - backbone: str - See backbones.py
        - warm_up_phase_len: int - defines the base lengh of the warm-up phase (default is 30).
        - joint_phase_len: int - defines the base lengh of the joint phase (default is 70).
        - phase_multiplier: int - scales the number of epochs while inversely scaling the learning rates (bigger phases, more epochs/passes of data, less learning rate).
        - latent_dim_multiplier_exp: int - expotential of 2 for the latent dimension of the prototype layer.
        - phase_relative_lr_multiplier: float - multipy the learning rate of each phase (applied with).
        - joint_project_interval: int - interval projects during joint phase (default is None).
        - num_milestones: int - within the milestone range of the joint epoch, space this many steps of the optimizer (impacted by effective learning rate).
        - num_addon_layers: int - number of addon layers.
        - activation_function: str - configure activation functions.
        - fa_func: str - one of "serial", "l2", or "square" to indicate which type of fine annotation loss to use. if None, fine annotation is deactivated.
        - fa_coef: float - coefficient for fine annotation loss term.
        - depth: int - max depth of the tree, creates 2^(d)-1 prototypes.
        - pruning_threshold: float - an internal node will be pruned when the maximum class probability in the distributions of all leaves below this node are lower than this threshold.
        - log_probabilities: bool - convert all tree probability calculations to log probabilities (useful for numerical stability)
        - disable_derivative_free_leaf_optim: bool - flag that optimizes the leafs with gradient descent when set instead of using the derivative-free algorithm.
        - disable_backbone_lastlayer_training: bool - disables training of last layer of the backbone during warm-up phase
        - k_for_topk: int - aggregates the top k activations for each prototype.
        - lr_step_gamma: float - configure the gamma of stochastic gradient descent .
        - dry_run: bool - configure the training run, but do not execute it.
        - verify: bool - configure a training run for a single epoch of all phases.
        - interpretable_metrics: bool - whether to calculate interpretable metrics.
        - save_on_proto_updates: bool - whether or not to save intermediate models to run pacmap on.
        - dataset: str - name of the dataset to use.
        - proto_channels: int -
        - rsample_init: bool - whether to initialize the prototypes are parts of the input images.
        - min_post_project_target_metric: float -
    """

    if verify:
        log.info("Running in verification mode to check training will run all phases")
        warm_up_phase_len = 1
        joint_phase_len = 1
        phase_multiplier = 1
        phase_relative_lr_multiplier = 1
        log_probabilities = True
        proto_channels = 256
        depth = 2
        num_addon_layers = 1
        save_on_proto_updates = False
        pruning_threshold = 0.01

    log.info("Training config %s", json.dumps(locals()))
    log.log(
        logging.INFO if dry_run else logging.DEBUG,
        f'Running prototree training as a {"dry_run" if dry_run else "full run"}',
    )

    setup = {
        "depth": depth,
        "num_prototypes": 2**depth - 1,
        "coefs": {
            "nll_loss": 1,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    post_forward_calculations = []

    # ProtoTree has no grad accumulation
    if "dense" in backbone:
        batch_sizes = {"train": 32, "project": 32, "val": 32}
    elif "convnext" in backbone:
        batch_sizes = {"train": 95 // 2, "project": 75 // 2, "val": 100 // 2}
    else:
        batch_sizes = {"train": 64, "project": 64, "val": 64}

    split_dataloaders = datasets.training_dataloaders(
        dataset,
        batch_sizes=batch_sizes,
        train_dir="train_corner_crop",
        val_dir="validation",
        project_dir="train_cropped",
        # FIXME this only makes sense for CUB200
        **({"augment": True} if dataset == "cub200_cropped" else {}),
    )

    if fa_coef is not None:
        post_forward_calculations = [
            BilinearUpsampleActivations(split_dataloaders.image_size)
        ]

    activation = lookup_activation(activation_function)

    backbone_module = construct_backbone(backbone)

    if backbone.startswith("resnet50"):
        # lop off the last layer as was done in prototree
        backbone_module.embedded_model.layer4 = torch.nn.Sequential(
            *list(backbone_module.embedded_model.layer4.children())[:-1]
        )

    add_on_layers = AddonLayers(
        num_prototypes=setup["num_prototypes"],
        input_channels=backbone_module.latent_dimension[0],
        proto_channel_multiplier=(
            2**latent_dim_multiplier_exp
            if latent_dim_multiplier_exp is not None
            else None
        ),
        proto_channels=proto_channels,
        num_addon_layers=num_addon_layers,
    )

    vppn = ProtoTree(
        backbone=backbone_module,
        add_on_layers=add_on_layers,
        activation=activation,
        num_classes=split_dataloaders.num_classes,
        depth=depth,
        log_probabilities=log_probabilities,
        k_for_topk=k_for_topk,
        pruning_threshold=pruning_threshold,
    )

    train_logger = WeightsAndBiasesTrainLogger(
        device=setup["device"],
        calculate_best_for=["accuracy"],
        wandb_plots=(
            ["pr", "roc", "conf_mat"] if split_dataloaders.num_classes <= 10 else None
        ),
    )

    loss = loss_for_coefficients(
        setup["coefs"],
        fa_func=fa_func,
    ).to(setup["device"])

    lastlayer_mapping = {
        "resnet50": "layer4.2",
        "vgg19": "features.34",
        "densenet161": "features.denseblock4.denselayer24",
        "squeezenet1_0": "features.12",
    }

    lastlayer_name = lastlayer_mapping[backbone.split("[", 1)[0]]

    # account for the number of epochs (with more epochs, reduce learning)
    # then adjust resulting learning rate by a hyperparameter mutlplier
    effective_lr_multiplier = (1 / float(phase_multiplier)) * float(
        phase_relative_lr_multiplier
    )

    log.info(f"Effective LR multiplier: {effective_lr_multiplier}")

    training_schedule = ProtoTreeTrainingSchedule(
        model=vppn,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=loss,
        num_warm_epochs=warm_up_phase_len * phase_multiplier,
        num_joint_epochs=joint_phase_len * phase_multiplier,
        joint_epochs_before_lr_milestones=(joint_phase_len // 2 - joint_phase_len // 12)
        * phase_multiplier,
        num_milestones=num_milestones,
        backbone_lr=1e-5 * effective_lr_multiplier,
        prototype_lr=0.001 * effective_lr_multiplier,
        add_on_lr=0.001 * effective_lr_multiplier,
        backbone_layers_override_lr=([lastlayer_name], 0.001 * effective_lr_multiplier),
        lr_weight_decay=weight_decay,
        lr_step_gamma=lr_step_gamma,
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
        )
