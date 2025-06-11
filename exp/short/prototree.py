import json
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Tuple

import numpy as np
import torch
from protopnet import datasets
from protopnet.activations import lookup_activation
from protopnet.backbones import construct_backbone, default_batch_sizes_for_backbone
from protopnet.embedding import AddonLayers
from protopnet.model_losses import loss_for_coefficients
from protopnet.models.prototree import (
    ProtoTree,
    ProtoTreeBackpropPhaseWithLeafUpdate,
    TrainProtoTreeLayersUsingProtoPNetNames,
)
from protopnet.prototypical_part_model import ProtoPNet
from protopnet.train.checkpointing import ModelCheckpointer
from protopnet.train.logging.types import TrainLogger
from protopnet.train.logging.weights_and_biases import WeightsAndBiasesTrainLogger
from protopnet.train.scheduling.early_stopping import SweepProjectEarlyStopping
from protopnet.train.scheduling.scheduling import (
    ClassificationInferencePhase,
    ProjectPhase,
    PrunePrototypesPhase,
    _NoGradPhaseMixin,
)
from protopnet.train.scheduling.types import (
    Phase,
    PostPhaseSummary,
    StepContext,
    TrainingSchedule,
)
from protopnet.train.trainer import MultiPhaseProtoPNetTrainer

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockProjectPhase(_NoGradPhaseMixin):
    name: str = "mock_project"

    def run_step(
        self, model: ProtoPNet, step_context: StepContext
    ) -> Optional[Dict[str, Any]]:
        return {}

    def after_training(
        self,
        model: ProtoPNet,
        metric_logger: TrainLogger,
        checkpointer: ModelCheckpointer,
        post_phase_summary: PostPhaseSummary,
    ) -> Optional[Dict[str, Any]]:
        """
        Clean up at the end of the phase.

        Args:
            model: The model being trained

        Returns:
            Optional dictionary of cleanup results
        """
        metric = post_phase_summary.final_target_metric
        metric_logger.log_metrics(
            "project",
            step=post_phase_summary.last_step + 1,
            prototypes_embedded_state=False,
            precalculated_metrics={metric[0]: metric[1]},
        )


class BracketedProtoTreeTrainingSchedule(TrainingSchedule):
    def __init__(
        self,
        model: ProtoTree,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        project_loader: torch.utils.data.DataLoader,
        loss: torch.nn.Module,
        num_warm_epochs: int = 30,
        num_joint_epochs: int = 70,
        joint_epochs_before_lr_milestones: int = 30,
        num_milestones: int = 5,
        backbone_lr: float = 1e-5,
        prototype_lr: float = 0.001,
        add_on_lr: float = 0.001,
        backbone_layers_override_lr: Optional[Tuple[Collection[str], float]] = None,
        lr_weight_decay: float = 0.0,
        lr_step_gamma: float = 0.5,
        adamw_weight_decay: float = 0.0,
        adamw_eps: float = 1e-07,
        phase_config_kwargs: Dict[str, Any] = {},
        num_accumulation_batches: int = 1,
    ):
        """
        Args:
            backbone_layer_lrs: If None, all backbone layers will have the same learning rate. Otherwise, a tuple of the form (lastlayer_name, lr) will set the learning rate for named layers.
        """

        if num_accumulation_batches > 1:
            warnings.warn(
                "ProtoTree tree updates happen every batch irrespective of grad accumulation. This may lead to unexpected behavior."
            )

        backbone_standard_params = []
        backbone_override_params = []

        if backbone_layers_override_lr and len(backbone_layers_override_lr[0]) > 0:
            backbones_to_override_train = backbone_layers_override_lr[0]
            for name, param in model.backbone.named_parameters():
                if any(
                    (override in name) for override in backbone_layers_override_lr[0]
                ):
                    backbone_override_params.append(param)
                else:
                    backbone_standard_params.append(param)
        else:
            backbones_to_override_train = None

        opt_conf_add_on = {
            "params": model.add_on_layers.parameters(),
            "lr": add_on_lr,
            "weight_decay": lr_weight_decay,
        }

        opt_conf_backbone = {
            "params": backbone_standard_params,
            "lr": backbone_lr,
            "weight_decay": lr_weight_decay,
        }

        opt_conf_prototypes = {
            "params": model.prototype_layer.parameters(),
            "lr": prototype_lr,
            "weight_decay": lr_weight_decay,
        }

        if len(backbone_override_params) > 0:
            maybe_opt_conf_backbone_overrides = [
                {
                    "params": backbone_override_params,
                    "lr": backbone_layers_override_lr[1],
                    "weight_decay": lr_weight_decay,
                }
            ]
        else:
            maybe_opt_conf_backbone_overrides = []

        backprop_shared_config = {
            "dataloader": train_loader,
            "num_accumulation_batches": num_accumulation_batches,
            "loss": loss,
            **phase_config_kwargs,
        }

        eval = ClassificationInferencePhase(
            name="eval", dataloader=val_loader, loss=loss, **phase_config_kwargs
        )

        warm = ProtoTreeBackpropPhaseWithLeafUpdate(
            name="warm",
            optimizer=torch.optim.AdamW(
                [opt_conf_add_on, opt_conf_prototypes]
                + maybe_opt_conf_backbone_overrides,
                eps=adamw_eps,
                weight_decay=adamw_weight_decay,
            ),
            set_training_layers_fn=TrainProtoTreeLayersUsingProtoPNetNames(
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
                backbone_overrides=backbones_to_override_train,
            ),
            **backprop_shared_config,
        )

        # *opt_conf_backbone_last_layer
        joint_optimizer = torch.optim.AdamW(
            [opt_conf_backbone, opt_conf_add_on, opt_conf_prototypes]
            + maybe_opt_conf_backbone_overrides,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        )
        lr_milestones = list(
            np.linspace(
                num_warm_epochs + joint_epochs_before_lr_milestones,
                num_warm_epochs + num_joint_epochs,
                num=num_milestones,
                dtype=int,
            )
        )
        joint = ProtoTreeBackpropPhaseWithLeafUpdate(
            name="joint",
            optimizer=joint_optimizer,
            set_training_layers_fn=TrainProtoTreeLayersUsingProtoPNetNames(
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
                backbone_overrides=backbones_to_override_train,
            ),
            scheduler=torch.optim.lr_scheduler.MultiStepLR(
                joint_optimizer,
                milestones=lr_milestones,
                gamma=lr_step_gamma,
            ),
            **backprop_shared_config,
        )

        log.info(f"Milestones (decay={lr_step_gamma}){lr_milestones}")

        prune = PrunePrototypesPhase()
        project = ProjectPhase(
            dataloader=project_loader,
        )

        bracket = MockProjectPhase()

        super().__init__(
            default_eval_phase=eval,
            phases=[
                Phase(train=warm, duration=max(1, 1 * num_warm_epochs // 3)),
                Phase(train=bracket, duration=1),
                Phase(train=warm, duration=max(1, 1 * num_warm_epochs // 3)),
                Phase(train=bracket, duration=1),
                Phase(train=warm, duration=max(1, 1 * num_warm_epochs // 3)),
                Phase(train=joint, duration=max(1, 1 * num_joint_epochs // 6)),
                Phase(train=bracket, duration=1),
                Phase(train=joint, duration=max(1, 1 * num_joint_epochs // 3)),
                Phase(train=bracket, duration=1),
                Phase(train=joint, duration=max(1, num_joint_epochs // 2)),
                Phase(train=prune),
                Phase(train=project),
                Phase(train=bracket, duration=1),
            ],
        )


def train(
    *,
    backbone="resnet50[pretraining=inaturalist]",
    warm_up_phase_len_at_lr1: int = 30,
    joint_phase_len_at_lr1: int = 70,
    backbone_lr_multiplier: float = 1.0,
    non_backbone_lr_multiplier: float = 1.0,
    num_lr_milestones: int = 5,
    num_addon_layers: int = 1,
    activation_function: str = "exp_l2",
    depth: int = 9,
    pruning_threshold: float = 0.01,
    log_probabilities: bool = False,
    k_for_topk: int = 1,
    lr_step_gamma: float = 0.5,
    dataset="cub200",
    proto_channels: int = 256,
    lr_weight_decay: float = 0.0,
):

    log.info("Training config %s", json.dumps(locals()))

    if backbone_lr_multiplier == 0.0:
        backbone_lr_multiplier = 0.05
    if non_backbone_lr_multiplier == 0.0:
        non_backbone_lr_multiplier = 0.05

    if lr_step_gamma == 0.0:
        lr_step_gamma = 0.1

    setup = {
        "depth": depth,
        "num_prototypes": 2**depth - 1,
        "coefs": {
            "nll_loss": 1,
        },
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    batch_sizes, num_accumulation_batches = default_batch_sizes_for_backbone(backbone)

    if dataset == "cub200_cropped":
        split_dataloaders = datasets.training_dataloaders(
            dataset,
            batch_sizes=batch_sizes,
            train_dir="train_corner_crop",
            val_dir="validation",
            project_dir="train_cropped",
            **({"augment": True} if dataset == "cub200_cropped" else {}),
        )
        phase_multiplier = 1
    else:
        split_dataloaders = datasets.training_dataloaders(
            dataset,
            batch_sizes=batch_sizes,
        )
        # update training length for online augmentation
        phase_multiplier = 5

    activation = lookup_activation(activation_function)

    backbone_module = construct_backbone(backbone)

    add_on_layers = AddonLayers(
        num_prototypes=setup["num_prototypes"],
        input_channels=backbone_module.latent_dimension[0],
        proto_channels=proto_channels,
        num_addon_layers=num_addon_layers,
    )

    prototree = ProtoTree(
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
    )

    protopnet_loss = loss_for_coefficients(
        setup["coefs"],
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

    def scale_int_by_lr(x, lr_multiplier):
        """
        Scale an integer by the lr multiplier, and return at least 1.

        Since the learning rate multipliers are log distributed near zero, the rate modification is cbrt(lr_multiplier).
        """
        return max(1, int(x / np.cbrt(lr_multiplier)))

    num_joint_epochs = (
        scale_int_by_lr(joint_phase_len_at_lr1, backbone_lr_multiplier)
        * phase_multiplier
    )

    schedule_args = dict(
        num_warm_epochs=scale_int_by_lr(
            warm_up_phase_len_at_lr1, non_backbone_lr_multiplier
        )
        * phase_multiplier,
        num_joint_epochs=num_joint_epochs,
        joint_epochs_before_lr_milestones=int(
            num_joint_epochs // 2 - num_joint_epochs // 12
        ),
        num_milestones=min(num_joint_epochs, num_lr_milestones),
        backbone_lr=1e-5 * backbone_lr_multiplier,
        prototype_lr=0.001 * non_backbone_lr_multiplier,
        add_on_lr=0.001 * non_backbone_lr_multiplier,
        backbone_layers_override_lr=(
            [lastlayer_name],
            0.001 * non_backbone_lr_multiplier,
        ),
        lr_weight_decay=lr_weight_decay,
        lr_step_gamma=lr_step_gamma,
    )
    log.info("Training schedule arguments %s", json.dumps(schedule_args))

    training_schedule = BracketedProtoTreeTrainingSchedule(
        model=prototree,
        train_loader=split_dataloaders.train_loader,
        val_loader=split_dataloaders.val_loader,
        project_loader=split_dataloaders.project_loader,
        loss=protopnet_loss,
        **schedule_args,
        num_accumulation_batches=num_accumulation_batches,
        phase_config_kwargs={
            "device": setup["device"],
        },
    )

    ppn_trainer = MultiPhaseProtoPNetTrainer(
        device=setup["device"],
        metric_logger=train_logger,
        num_classes=split_dataloaders.num_classes,
    )

    early_stopping = SweepProjectEarlyStopping(
        metric_name="accuracy",
        min_runs_required=8,
        percentile_threshold=50.0,
        # see the bracketed phase in the custom training schedule
        brackets=[
            0,
            1,
            2,
            3,
        ],
    )

    ppn_trainer.train(
        model=prototree,
        training_schedule=training_schedule,
        target_metric_name="accuracy",
        early_stopping=early_stopping,
    )
