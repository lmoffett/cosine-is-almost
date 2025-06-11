from typing import Any, Dict, Optional

import torch

from protopnet.train.metrics import TrainingMetrics

from ..embedding import EmbeddedBackbone
from ..prediction_heads import LinearClassPrototypePredictionHead
from ..prototype_layers import ClassAwarePrototypeLayer
from ..prototypical_part_model import ProtoPNet
from ..train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    TrainLayersUsingProtoPNetNames,
)
from ..train.scheduling.types import IterativePhase, Phase, TrainingSchedule


class VanillaProtoPNet(ProtoPNet):
    def __init__(
        self,
        backbone: EmbeddedBackbone,
        add_on_layers,
        activation,
        num_classes: int,
        num_prototypes_per_class: int,
        k_for_topk: int = 1,
        **kwargs,
    ):
        num_prototypes = num_classes * num_prototypes_per_class

        # TODO: SHOULD BE CALLED FROM SAME INFO AS SELF.prototype_meta
        prototype_class_identity = torch.zeros(num_prototypes, num_classes)

        for j in range(num_prototypes):
            prototype_class_identity[j, j // num_prototypes_per_class] = 1

        latent_channels = add_on_layers.proto_channels

        prototype_layer = ClassAwarePrototypeLayer(
            activation_function=activation,
            latent_channels=latent_channels,
            class_specific_project=True,
            prototype_class_identity=prototype_class_identity,
            k_for_topk=k_for_topk,
        )

        prediction_head = LinearClassPrototypePredictionHead(
            prototype_class_identity=prototype_class_identity,
            k_for_topk=k_for_topk,
        )

        super(VanillaProtoPNet, self).__init__(
            backbone=backbone,
            add_on_layers=add_on_layers,
            activation=activation,
            prototype_layer=prototype_layer,
            prototype_prediction_head=prediction_head,
            k_for_topk=k_for_topk,
            **kwargs,
        )


class ProtoPNetTrainingSchedule(TrainingSchedule):
    def __init__(
        self,
        *,
        model: ProtoPNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        project_loader: torch.utils.data.DataLoader,
        loss: torch.nn.Module,
        last_only_loss: Optional[torch.nn.Module] = None,
        num_accumulation_batches: int = 1,
        add_on_lr: float = 0.003,
        num_warm_epochs: int = 10,
        num_preproject_joint_epochs: int = 10,
        backbone_lr: float = 1e-4,
        prototype_lr: float = 0.003,
        pred_head_lr: float = 1e-4,
        weight_decay: float = 1e-3,
        joint_lr_step_size: int = 5,
        joint_lr_step_gamma: float = 0.1,
        last_only_epochs_per_project: int = 20,
        joint_epochs_per_phase: int = 10,
        post_project_phases: int = 12,
        phase_config_kwargs: Dict[str, Any] = {},
        extra_training_metrics: Optional[TrainingMetrics] = None,
    ):
        """
        Args:
            - model: model class object
            - train_loader: dataloader of the training set
            - ...
            - loss: loss terms
            - last_only_loss: a loss term that applies only the the last layers (prediction head)
            - num_accumulation_batches: how many batches in gradient accumulation
            - add_on_lr: learning rate of the addon layers
            - num_warm_epochs: warm epochs in original protopnet freeze the backbone+predictionhead, train only the prototype layers
            - num_preproject_joint_epochs: how many joint (end-to-end) training epochs before the first project phase
            - ...
            - joint_lr_step_size: number of epochs for each learning rate step down
            - last_only_epochs_per_project: epochs after project where most features are frozen, only the prediction head trains
            - joint_epochs_per_phase: number of joint (end-to-end) training epochs before each projection step after warmups
            - post_project_phases: how many times to loop (last-only epochs -> joint (end-to-end) epochs -> project epochs) after the first project
        """
        if last_only_loss is None:
            last_only_loss = loss

        eval = ClassificationInferencePhase(
            name="eval",
            dataloader=val_loader,
            loss=loss,
            extra_training_metrics=extra_training_metrics,
            **phase_config_kwargs,
        )

        backprop_shared_config = {
            "dataloader": train_loader,
            "num_accumulation_batches": num_accumulation_batches,
            "loss": loss,
            "extra_training_metrics": extra_training_metrics,
            **phase_config_kwargs,
        }

        opt_conf_add_on = {
            "params": model.add_on_layers.parameters(),
            "lr": add_on_lr,
            "weight_decay": weight_decay,
        }

        opt_conf_proto = {
            "params": model.prototype_layer.prototype_tensors,
            "lr": prototype_lr,
        }

        opt_conf_backbone = {
            "params": model.backbone.parameters(),
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        }

        opt_conf_head = {
            # FIXME - shouldn't this be model.prototype_prediction_head.parameters()?
            "params": model.prototype_prediction_head.class_connection_layer.parameters(),
            "lr": pred_head_lr,
        }

        warm = ClassificationBackpropPhase(
            name="warm",
            optimizer=torch.optim.Adam([opt_conf_add_on, opt_conf_proto]),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
            ),
            **backprop_shared_config,
        )

        joint_optimizer = torch.optim.Adam(
            [opt_conf_add_on, opt_conf_backbone, opt_conf_proto]
        )

        joint = ClassificationBackpropPhase(
            name="joint",
            optimizer=joint_optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size,
                gamma=joint_lr_step_gamma,
            ),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
            ),
            **backprop_shared_config,
        )

        last_only_shared_config = backprop_shared_config.copy()
        last_only_shared_config["loss"] = last_only_loss
        last_only = ClassificationBackpropPhase(
            name="last_only",
            optimizer=torch.optim.Adam([opt_conf_head]),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=False,
                train_add_on_layers=False,
                train_prototype_layer=False,
                train_prototype_prediction_head=True,
            ),
            **last_only_shared_config,
        )

        project = ProjectPhase(
            dataloader=project_loader,
        )

        maybe_iterative_phase = (
            [
                IterativePhase(
                    phases=[
                        Phase(train=last_only, duration=last_only_epochs_per_project),
                        Phase(train=joint, duration=joint_epochs_per_phase),
                        Phase(train=project),
                    ],
                    iterations=post_project_phases,
                )
            ]
            if post_project_phases > 0
            else []
        )

        super().__init__(
            default_eval_phase=eval,
            phases=[
                Phase(train=warm, duration=num_warm_epochs),
                Phase(train=joint, duration=num_preproject_joint_epochs),
                Phase(train=project),
                *maybe_iterative_phase,
                Phase(train=last_only, duration=last_only_epochs_per_project),
            ],
        )
