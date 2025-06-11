import logging
import warnings
from dataclasses import dataclass
from typing import Any, Collection, Dict, Optional, Tuple

import numpy as np
import torch

from protopnet.train.metrics import TrainingMetrics

from ..embedding import EmbeddedBackbone
from ..prediction_heads import ProtoTreePredictionHead
from ..prototype_layers import PrototypeLayer
from ..prototypical_part_model import ProtoPNet
from ..train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    PrunePrototypesPhase,
    TrainLayersUsingProtoPNetNames,
)
from ..train.scheduling.types import Phase, StepContext, TrainingSchedule

log = logging.getLogger(__name__)


class ProtoTree(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,
        num_classes: int,
        depth: int,
        log_probabilities: bool,
        warn_on_errors: bool = False,
        k_for_topk: int = 1,
        pruning_threshold: float = 0.01,
    ):
        num_prototypes = (2**depth) - 1

        latent_channels = add_on_layers.proto_channels

        prototype_layer = PrototypeLayer(
            num_prototypes=num_prototypes,
            activation_function=activation,
            latent_channels=latent_channels,
            k_for_topk=k_for_topk,
        )

        prediction_head = ProtoTreePredictionHead(
            num_classes=num_classes,
            depth=depth,
            log_probabilities=log_probabilities,
            k_for_topk=k_for_topk,
            pruning_threshold=pruning_threshold,
            disable_derivative_free_leaf_optim=False,
        )

        super(ProtoTree, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
        )

        # re-init prototype tensors and reinit addon layers
        self.reinit_parameters()

    def batch_derivative_free_update(self, output, target, num_batches, old_leaf_dist):
        """
        If derivative free update is enabled, this method uses the
        output from the current batch (and its associated targets)
        to update the tree leaf distributions. Please view trainer
        for more context.

        Args:
            - output - last forward result for current training batch
            - target - ground truth target of current training batch
            - num_batches - number of batches captured from the dataloader
            - old_dist_params - distributions at each leaf in previous iteration
        """
        self.prototype_prediction_head.batch_derivative_free_tree_update(
            output, target, num_batches, old_leaf_dist
        )

    def reinit_parameters(self):
        """
        Implementing from https://github.com/M-Nauta/ProtoTree/blob/789be8545af34539f418fc3d4d8e473015022da8/util/init.py#L64-L65
        """

        def init_weights_xavier(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(
                    m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
                )

        mean = 0.5
        std = 0.1

        with torch.no_grad():
            torch.nn.init.normal_(self.prototype_layer.prototype_tensors, mean, std)
            self.add_on_layers.apply(init_weights_xavier)

    def prune_prototypes(self):
        """
        logic to identify which nodes to prune
        """

        # prune tree - this mutates the tree structures but does not update
        # mapping between branches and prototypes because tree model does
        # not natively prune the prototype layer (calculates distance for
        # pruned prototypes)
        prune_prototype_indices = self.prototype_prediction_head.prune_tree()

        # if any prototypes were pruned from tree, selected remaining prototype tensors
        if len(prune_prototype_indices):
            self.prototype_layer.prune_prototypes_by_index(prune_prototype_indices)


class ProtoTreeTrainingSchedule(TrainingSchedule):
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
        extra_training_metrics: Optional[TrainingMetrics] = None,
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
            "extra_training_metrics": extra_training_metrics,
            **phase_config_kwargs,
        }

        eval = ClassificationInferencePhase(
            name="eval",
            dataloader=val_loader,
            loss=loss,
            extra_training_metrics=extra_training_metrics,
            **phase_config_kwargs,
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

        super().__init__(
            default_eval_phase=eval,
            phases=[
                Phase(train=warm, duration=num_warm_epochs),
                Phase(train=joint, duration=num_joint_epochs),
                Phase(train=prune),
                Phase(train=project),
            ],
        )


@dataclass(frozen=True)
class TrainProtoTreeLayersUsingProtoPNetNames(TrainLayersUsingProtoPNetNames):
    backbone_overrides: Optional[Collection[str]] = None

    def overwrite_grad_epoch_settings(self, name, param, setting_attr, should_train):
        if (
            (setting_attr == "train_backbone")
            # TODO: this is unnecessary. If should_train is true, we should always train
            and (not should_train)
            and self.backbone_overrides
            and any(override in name for override in self.backbone_overrides)
        ):
            # during warmup, should_train is false but we need to train the last layer
            param.requires_grad = True


class ProtoTreeBackpropPhaseWithLeafUpdate(ClassificationBackpropPhase):
    """
    Phase that updates the leaf nodes of the tree model after each batch.
    """

    def _set_leaf_dist(self, model: ProtoTree):
        with torch.no_grad():
            self.old_leaf_dist = {
                leaf: leaf._dist_params.detach().clone()
                for leaf in model.prototype_prediction_head.prototree.leaves
            }

    def _pre_dataloader_step_init(self, model: ProtoTree, step_context: StepContext):
        """
        Initialize the optimizer for the current batch and capture the current state of the leaf nodes.
        """
        super()._pre_dataloader_step_init(model=model, step_context=step_context)
        if hasattr(self, "optimizer"):
            self._set_leaf_dist(model)

    def _pre_metrics_update(
        self,
        model: ProtoPNet,
        batch_data_dict: Dict[str, Any],
        output: Dict[str, Any],
        **kwargs,
    ):
        super()._pre_metrics_update(
            model=model, batch_data_dict=batch_data_dict, output=output, **kwargs
        )
        if hasattr(self, "optimizer"):
            original_state = model.training
            try:
                model.eval()
                with torch.no_grad():
                    model.batch_derivative_free_update(
                        output,
                        batch_data_dict["target"],
                        len(self.dataloader),
                        self.old_leaf_dist,
                    )
            finally:
                model.train(original_state)
