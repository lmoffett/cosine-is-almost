from typing import Dict, Tuple

import torch
import torch.nn as nn

from .losses import (
    ClosenessLoss,
    ClusterCost,
    CrossEntropyCost,
    DiscriminationLoss,
    FineAnnotationCost,
    GrassmannianOrthogonalityLoss,
    L1CostClassConnectionLayer,
    LossTerm,
    NegativeLogLikelihoodCost,
    OrthogonalityLoss,
    SeparationCost,
)
from .prototypical_part_model import ProtoPNet


class LinearBatchLoss(nn.Module):
    def __init__(self, batch_losses: list = [], device="cpu"):
        super(LinearBatchLoss, self).__init__()
        self.batch_losses = batch_losses
        self.device = device

    def required_forward_results(self):
        return {
            req
            for loss_component in self.batch_losses
            for req in loss_component.loss.required_forward_results
        }

    def forward(self, **kwargs):
        # Metrics dict comes from kwargs
        loss_term_dict = {}

        total_loss = torch.tensor(0.0, device=self.device)

        for loss_component in self.batch_losses:
            # Get args for loss from just the loss_component.required_forward_results from kwargs
            current_loss_args = {
                req: kwargs[req] for req in loss_component.loss.required_forward_results
            }

            current_loss_without_weight = loss_component.loss(**current_loss_args)

            # assert loss_component is a float
            current_loss = current_loss_without_weight * loss_component.coefficient

            loss_term_dict[loss_component.loss.name] = (
                current_loss_without_weight.item()
            )
            loss_term_dict[loss_component.loss.name + "_weighted"] = current_loss.item()
            loss_term_dict[loss_component.loss.name + "_coef"] = (
                loss_component.coefficient
            )

            total_loss += current_loss

        return total_loss, loss_term_dict

    def to(self, device):
        """
        Move the loss to the given device.
        """
        super().to(device)
        self.device = device
        for loss_component in self.batch_losses:
            if isinstance(loss_component.loss, nn.Module):
                loss_component.loss.to(device)
        return self

    def __repr__(self):
        return f"LinearBatchLoss(batch_losses={self.batch_losses})"


class LinearModelRegularization(nn.Module):
    def __init__(self, model_losses: list = [], device="cpu"):
        super(LinearModelRegularization, self).__init__()
        self.model_losses = model_losses
        self.device = device

    def forward(self, model: ProtoPNet, **kwargs):
        loss_term_dict = {}

        # TODO: Set device to be same as model based variables
        total_loss = torch.tensor(0.0, device=self.device)  # Adjust device as needed

        for loss_component in self.model_losses:
            current_loss_without_weight = loss_component.loss(model)
            current_loss = current_loss_without_weight * loss_component.coefficient

            loss_term_dict[loss_component.loss.name] = (
                current_loss_without_weight.item()
            )
            loss_term_dict[loss_component.loss.name + "_weighted"] = current_loss.item()
            loss_term_dict[loss_component.loss.name + "_coef"] = (
                loss_component.coefficient
            )

            total_loss += current_loss

        return total_loss, loss_term_dict

    def to(self, device):
        """
        Move the loss to the given device.
        """
        super().to(device)
        self.device = device
        for loss_component in self.model_losses:
            if isinstance(loss_component.loss, nn.Module):
                loss_component.loss.to(device)
        return self

    def __repr__(self):
        return f"LinearModelRegularization(model_losses={self.model_losses})"


class ProtoPNetLoss(nn.Module):
    def __init__(self, batch_losses, model_losses, device="cpu"):
        super(ProtoPNetLoss, self).__init__()

        self.batch_loss = LinearBatchLoss(batch_losses, device)
        self.model_regularization = LinearModelRegularization(model_losses, device)

        self.batch_loss_required_forward_results = (
            self.batch_loss.required_forward_results()
        )

    def forward(
        self,
        target: torch.Tensor,
        model: ProtoPNet,
        **kwargs,
    ) -> Tuple[torch.tensor, Dict[str, float]]:
        """
        Calculate batch loss and model regularization terms for a ProtoPNet model.

        Args:
            target (torch.Tensor): Groud truth labels for input batch.
            model (ProtoPNet): ProtoPNet model.

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - The first element of the tuple is the total computed loss (torch.Tensor).
                - The second element is a dictionary containing related loss term
                    calculations {(loss.name -> value), (loss.name+"_coef" -> value),
                    (loss.name+"_weighted" -> value)}

        Notes:
            - kwargs may contain additional arguments required for loss calculations
        """

        # dict for saving loss term calculations and related metrics
        output_dict = {}

        # Pass in all arguments to batch_loss
        batch_loss, loss_term_dict = self.batch_loss(
            target=target,
            **kwargs,
        )

        output_dict.update(loss_term_dict)

        # batch_loss = self.batch_loss(pred, target, similarity_score_to_each_prototype, upsampled_activation, prototypes_of_correct_class, prototypes_of_wrong_class, metrics_dict)
        model_regularization, loss_term_dict = self.model_regularization(model)

        output_dict.update(loss_term_dict)

        return (batch_loss + model_regularization, output_dict)

    def to(self, device):
        """
        Move the loss to the given device.
        """
        super().to(device)
        self.batch_loss.to(device)
        self.model_regularization.to(device)
        return self

    def __repr__(self):
        return f"ProtoPNetLoss(batch_losses={self.batch_loss}, model_losses={self.model_regularization})"


def loss_for_coefficients(
    coefs: Dict[str, float],
    class_specific_cluster: bool = True,
    ortho_p_norm: int = 2,
    grassmannian_orthogonality_loss_normalize: bool = True,
    grassmannian_orthogonality_loss_mini_batch_size: int = None,
    fa_func: str = "serial",
) -> ProtoPNetLoss:
    """
    Create a ProtoPNetLoss object with the given coefficients. Currently, there is no alternative to this function to create custom loss terms.
    """

    batch_losses = []

    if coefs.get("cross_entropy", False):
        batch_losses.append(
            LossTerm(loss=CrossEntropyCost(), coefficient=coefs["cross_entropy"])
        )

    if coefs.get("nll_loss", False):
        batch_losses.append(
            LossTerm(loss=NegativeLogLikelihoodCost(), coefficient=coefs["nll_loss"])
        )

    if coefs.get("cluster", False):
        batch_losses.append(
            LossTerm(
                loss=ClusterCost(class_specific=class_specific_cluster),
                coefficient=coefs["cluster"],
            )
        )

    if coefs.get("separation", False):
        batch_losses.append(
            LossTerm(loss=SeparationCost(), coefficient=coefs["separation"])
        )

    if coefs.get("fa", False):
        batch_losses.append(
            LossTerm(
                loss=FineAnnotationCost(fa_loss=fa_func),
                coefficient=coefs["fa"],
            )
        )

    model_losses = []

    if coefs.get("l1", False):
        model_losses.append(
            LossTerm(loss=L1CostClassConnectionLayer(), coefficient=coefs["l1"])
        )

    if "orthogonality_loss" in coefs:
        model_losses.append(
            LossTerm(
                loss=OrthogonalityLoss(p=ortho_p_norm),
                coefficient=coefs["orthogonality_loss"],
            )
        )

    if "grassmannian_orthogonality_loss" in coefs:
        model_losses.append(
            LossTerm(
                loss=GrassmannianOrthogonalityLoss(
                    normalize=grassmannian_orthogonality_loss_normalize,
                    mini_batch_size=grassmannian_orthogonality_loss_mini_batch_size,
                ),
                coefficient=coefs["grassmannian_orthogonality_loss"],
            )
        )

    if coefs.get("closeness_loss", False):
        model_losses.append(
            LossTerm(
                loss=ClosenessLoss(
                    class_specific=class_specific_cluster,
                ),
                coefficient=coefs["closeness_loss"],
            )
        )

    if coefs.get("discrimination_loss", False):
        model_losses.append(
            LossTerm(
                loss=DiscriminationLoss(
                    class_specific=class_specific_cluster,
                ),
                coefficient=coefs["discrimination_loss"],
            )
        )

    return ProtoPNetLoss(batch_losses=batch_losses, model_losses=model_losses)
