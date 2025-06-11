import warnings
from dataclasses import dataclass
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prototypical_part_model import ProtoPNet


class ClassAwareExtraCalculations:
    def __call__(
        self,
        *,
        target: torch.Tensor,
        model: ProtoPNet,
        similarity_score_to_each_prototype: torch.Tensor,
    ):
        """
        Calculates class-aware metrics using prototype similarities and class identities.

        This function was created to calculate metrics that require multiple forward passes through backbone. None currently implemented.

        Args:
            target (torch.Tensor): The target class labels for the batch.
            model: A ProtoPNet model instance that contains prototype_class_identity
                  and get_prototype_class_identity method.
            similarity_score_to_each_prototype (torch.Tensor): Similarity scores between
                input images and each prototype. Batch x Prototypes.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - prototype_class_identity: A one-hot encoded matrix of shape (num_prototypes, num_classes)
                  where each row represents a prototype and each column represents a class.

                - prototypes_of_correct_class: A binary tensor of shape (batch_size, num_prototypes)
                  where 1 indicates that a prototype belongs to the same class as the target
                  label for that batch item. For example, if batch item i has target class c,
                  then prototypes_of_correct_class[i,j] = 1 if prototype j belongs to class c.

                - prototypes_of_wrong_class: A binary tensor of shape (batch_size, num_prototypes)
                  that is the complement of prototypes_of_correct_class. A value of 1 indicates
                  that a prototype belongs to a different class than the target label for that
                  batch item.

                - incorrect_class_prototype_activations: A tensor of shape (batch_size,) containing
                  the highest similarity score achieved by any prototype belonging to an incorrect
                  class for each batch item. This can be used to measure how strongly an input
                  activates prototypes from classes other than its target class.

        Note:
            Returns an empty dictionary if the model doesn't have prototype_class_identity
            attribute.
        """

        extra_returns = {}
        # returns around prototype class identity
        if hasattr(model.prototype_layer, "prototype_class_identity"):
            prototype_class_identity = model.prototype_layer.prototype_class_identity

            extra_returns["prototype_class_identity"] = prototype_class_identity

            prototypes_of_correct_class = torch.t(
                model.get_prototype_class_identity(target)
            )
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            extra_returns["prototypes_of_correct_class"] = prototypes_of_correct_class
            extra_returns["prototypes_of_wrong_class"] = prototypes_of_wrong_class

            # FIXME: this is a prototype-to-sample similarity score calculation, implicitly done as a max
            # this should be renamed and deprecated to be replaced with a more general implemenation
            incorrect_class_prototype_activations, _ = torch.max(
                similarity_score_to_each_prototype * prototypes_of_wrong_class, dim=1
            )

            extra_returns["incorrect_class_prototype_activations"] = (
                incorrect_class_prototype_activations
            )

        return extra_returns


@dataclass
class LossTerm:
    loss: nn.Module
    coefficient: Union[Callable, float]


class CrossEntropyCost(nn.Module):
    def __init__(self):
        super(CrossEntropyCost, self).__init__()
        self.name = "cross_entropy"

        # TODO: Should these be functions or lists?
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        cross_entropy = torch.nn.functional.cross_entropy(logits, target)
        return cross_entropy


class NegativeLogLikelihoodCost(nn.Module):
    def __init__(self):
        super(NegativeLogLikelihoodCost, self).__init__()
        self.name = "nll_loss"

        # TODO: Should these be functions or lists?
        self.required_forward_results = {"logits", "target"}

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        nll_loss = torch.nn.functional.nll_loss(logits, target)
        return nll_loss


class L1CostClassConnectionLayer(nn.Module):
    """Calculates L1 norm of the class connection layer weights.

    Args:
        negative_classes_only (bool, optional): If True, only between class weights
            are considered. Class identity mask applied. Defaults to True.
    """

    def __init__(self, negative_classes_only: bool = True):
        super(L1CostClassConnectionLayer, self).__init__()
        self.name = "l1"
        self.negative_classes_only = negative_classes_only

    def forward(self, model: "ProtoPNet"):
        if self.negative_classes_only:
            weights = model.prototype_prediction_head.class_connection_layer.weight * (
                1 - model.prototype_prediction_head.prototype_class_identity.T
            )
        else:
            weights = model.prototype_prediction_head.class_connection_layer.weight

        return weights.norm(p=1)


class ClusterCost(nn.Module):
    def __init__(self, class_specific: bool = True):
        super(ClusterCost, self).__init__()
        self.class_specific = class_specific
        self.name = "cluster"

        self.required_forward_results = {
            "similarity_score_to_each_prototype",
            "prototypes_of_correct_class",
        }

    def forward(
        self, similarity_score_to_each_prototype, prototypes_of_correct_class=None
    ):
        # Raise Assertion if similarity_score_to_each_prototype, prototypes_of_correct_class is 1D
        assert similarity_score_to_each_prototype.dim() > 1 and (
            prototypes_of_correct_class is None or prototypes_of_correct_class.dim() > 1
        ), "Max activations or prototypes of correct class is 1D."

        if self.class_specific:
            assert (
                prototypes_of_correct_class is not None
            ), "Prototypes of correct class must be provided to calculate cluster cost."

            # correct_class_prototype_activations
            closest_sample_activations, _ = torch.max(
                similarity_score_to_each_prototype * prototypes_of_correct_class, dim=1
            )
        else:
            closest_sample_activations, _ = torch.max(
                similarity_score_to_each_prototype, dim=1
            )

        cluster_cost = torch.mean(closest_sample_activations)

        return cluster_cost


class SeparationCost(nn.Module):
    def __init__(self):
        super(SeparationCost, self).__init__()
        self.name = "separation"

        self.required_forward_results = {"incorrect_class_prototype_activations"}

    def forward(self, incorrect_class_prototype_activations):
        if incorrect_class_prototype_activations is None:
            raise ValueError(
                "Incorrect class prototype activations must be provided to calculate separation cost"
            )

        separation_cost = torch.mean(incorrect_class_prototype_activations)

        return separation_cost


class AverageSeparationCost(nn.Module):
    def __init__(self):
        super(AverageSeparationCost, self).__init__()
        self.name = "average_separation"

        self.required_forward_results = {
            "incorrect_class_prototype_activations",
            "prototypes_of_wrong_class",
        }

    def forward(
        self,
        incorrect_class_prototype_activations,
        prototypes_of_wrong_class=None,
    ):
        # Raise Assertion if prototypes_of_wrong_class is 1D
        assert prototypes_of_wrong_class.dim() > 1, "Prototypes of wrong class is 1D."

        if not (
            incorrect_class_prototype_activations is not None
            and prototypes_of_wrong_class is not None
        ):
            return None

        avg_separation_cost = incorrect_class_prototype_activations / torch.sum(
            prototypes_of_wrong_class, dim=1
        )

        avg_separation_cost = torch.mean(avg_separation_cost)

        return avg_separation_cost


class OffsetL2Cost(nn.Module):
    def __init__(self):
        super(OffsetL2Cost, self).__init__()

        self.name = "offset_l2"

    def forward(self, input_normalized: torch.Tensor, model: "ProtoPNet"):
        # Need to pass in input_normalized

        # TODO: Need conv_offset Sequential in model for this to work
        # offsets = model.module.conv_offset(input_normalized)
        offsets = torch.ones_like(input_normalized)

        offset_l2 = offsets.norm()
        return offset_l2


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality loss for the prototype tensors.

    This loss encourages the prototype tensors to be orthogonal to each other.

    Args:
        p (int, optional): The norm to use for the loss. Defaults to 2.
        mode (str, optional): Whether loss is calculated as "intra" or "inter" class orthogonality.
            Defaults to "intra".
    """

    def __init__(self, p=2, mode="intra"):
        super(OrthogonalityLoss, self).__init__()
        self.p = p
        self.mode = mode
        assert self.mode in ["intra", "inter"]

        if self.mode == "inter":
            warnings.warn(
                "Orthogonality loss is set to 'inter' mode. This mode is not well understood."
            )

        self.name = "orthogonality_loss"

    def forward(self, model: "ProtoPNet"):
        # Grab our prototype tensors, of shape (num_protos, channel, proto_h, proto_w)
        prototype_tensors = model.prototype_layer.prototype_tensors

        # Seperate prototypes out by class
        prototype_tensors = prototype_tensors.reshape(
            model.prototype_layer.num_prototypes_per_class,
            model.prototype_layer.num_classes,
            *prototype_tensors.shape[-3:],
        )

        # Set shape based on mode
        # FIXME: The inter permute reshape mixes the prototypes between classes,
        # specifically, all index 1 prototypes move to the same class, all index 2
        # prototypes move to the same class, etc. This behavior may not be
        # anticipated and further discussion needed. TLDR - inter mode is not understood.
        permute_shape = [0, 1, 3, 4, 2] if self.mode == "inter" else [1, 0, 3, 4, 2]

        # Permute and reshape these to (num_classes, protos_per_class*parts_per_proto, channel)
        prototype_tensors = prototype_tensors.permute(*permute_shape).reshape(
            model.prototype_layer.num_classes, -1, prototype_tensors.shape[-3]
        )

        # Normalize each part to unit length
        prototype_tensors = F.normalize(prototype_tensors, p=2, dim=-1)

        # Get our (num_classes, protos_per_class*parts_per_proto, protos_per_class*parts_per_proto)
        # orthogonality matrix
        orthogonalities = torch.bmm(
            prototype_tensors, prototype_tensors.transpose(-2, -1)
        )

        # Subtract out the identity matrix
        orthogonalities = orthogonalities - torch.eye(
            orthogonalities.shape[-1], device=orthogonalities.device
        ).unsqueeze(0)

        # And compute our loss
        ortho_loss = torch.sum(torch.norm(orthogonalities, p=self.p, dim=(1, 2)))

        return ortho_loss


class GrassmannianOrthogonalityLoss(nn.Module):
    """
    Calculates the Grassmannian Orthogonality loss between projection matrices.

    In the case of ProtoPNet, the subspaces are class spaces spanned by the class
    prototype basis vectors. This is described in the TesNet paper.

    Reference: https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf

    Args:
        normalize (bool, optional): Whether to normalize the prototype tensors before computing the loss. Defaults to True.
        mini_batch_size (int, optional): Mini batch size for computing the loss. Defaults to None.
    """

    def __init__(self, normalize: bool = True, mini_batch_size: int = None):
        super(GrassmannianOrthogonalityLoss, self).__init__()
        self.name = "grassmannian_orthogonality_loss"
        self.normalize = normalize
        self.mini_batch_size = mini_batch_size

        if not self.normalize:
            warnings.warn(
                "Prototype tensors are not normalized. This may affect the loss calculation."
                "Grassmann chordal distance may not be well-defined."
            )

        warnings.warn(
            "This loss will be positive if the prototypes are similar (aligned), zero if the prototypes are orthogonal, "
            "as a result the hyperparameter associated with this loss should be negative to maximize the distance between prototypes."
        )

    def _reshape_and_normalize_prototypes(self, model: "ProtoPNet"):
        """Reshapes and normalizes the prototype tensors by class.

        Args:
            model (ProtoPNet): The ProtoPNet model containing the prototype tensors.

        Returns:
            torch.Tensor: The reshaped and optionally normalized prototype tensors.
        """
        prototype_tensors = model.prototype_layer.prototype_tensors

        # Reshape to separate prototypes by class
        prototype_tensors = prototype_tensors.reshape(
            model.prototype_layer.num_classes,
            model.prototype_layer.num_prototypes_per_class,
            *prototype_tensors.shape[-3:],
        )

        # Permute and reshape to (num_classes, num_vectors, channels)
        prototype_tensors = prototype_tensors.permute(0, 1, 3, 4, 2).reshape(
            model.prototype_layer.num_classes, -1, prototype_tensors.shape[-3]
        )

        # Normalize each vector to unit length (L2 normalization)
        if self.normalize:
            prototype_tensors = F.normalize(prototype_tensors, p=2, dim=-1)
        else:
            prototype_tensors = prototype_tensors

        return prototype_tensors

    def _compute_loss_with_minibatch(self, projection_matrices, model):
        """Calculates the Grassmannian Orthogonality loss using mini-batching.

        Args:
            projection_matrices (torch.Tensor): Projection matrices for each class.
            model (ProtoPNet): The model containing the prototype layer info.

        Returns:
            torch.Tensor: The computed Grassmannian Orthogonality loss.
        """
        total_loss = 0.0
        total_classes = model.prototype_layer.num_classes
        num_batches = (total_classes + self.mini_batch_size - 1) // self.mini_batch_size

        device = projection_matrices.device
        dtype = projection_matrices.dtype

        projection_matrices_flat = projection_matrices.view(total_classes, -1)

        for batch_i in range(num_batches):
            start_i, end_i = self._get_batch_indices(batch_i, total_classes)
            batch_i_matrices = projection_matrices_flat[start_i:end_i]

            for batch_j in range(batch_i, num_batches):
                start_j, end_j = self._get_batch_indices(batch_j, total_classes)
                batch_j_matrices = projection_matrices_flat[start_j:end_j]

                # Compute pairwise differences between batches
                differences = batch_i_matrices.unsqueeze(
                    1
                ) - batch_j_matrices.unsqueeze(0)

                # Compute Frobenius norms of differences
                distances = torch.norm(differences, dim=2, p=2)

                # Create mask
                if batch_i == batch_j:
                    mask = torch.triu(torch.ones_like(distances), diagonal=1)
                else:
                    mask = torch.ones_like(distances)
                mask = mask.to(dtype=dtype, device=device)

                # Apply mask
                masked_distances = distances * mask

                # Sum the distances
                total_loss += masked_distances.sum()

        # Apply scaling factor
        scaling_factor = 1 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        total_loss = scaling_factor * total_loss

        return total_loss

    def _compute_loss_without_minibatch(self, projection_matrices, model):
        """Calculates the Grassmannian Orthogonality loss without mini-batching.

        Args:
            projection_matrices (torch.Tensor): Projection matrices for each class.
            model (ProtoPNet): The model containing the prototype layer info.

        Returns:
            torch.Tensor: The computed Grassmannian Orthogonality loss.
        """
        num_classes = projection_matrices.size(0)
        device = projection_matrices.device

        # Flatten projection matrices
        projection_matrices_flat = projection_matrices.view(num_classes, -1)

        # Compute pairwise differences
        differences = projection_matrices_flat.unsqueeze(
            1
        ) - projection_matrices_flat.unsqueeze(0)

        # Compute Frobenius norms of differences
        distances = torch.norm(differences, dim=2, p=2)

        # Create mask for upper triangular matrix without diagonal
        mask = torch.triu(torch.ones_like(distances), diagonal=1)
        mask = mask.to(dtype=distances.dtype, device=device)

        # Apply mask
        masked_distances = distances * mask

        # Compute total loss
        scaling_factor = 1 / torch.sqrt(
            torch.tensor(2.0, dtype=distances.dtype, device=device)
        )
        total_loss = scaling_factor * masked_distances.sum()

        return total_loss

    def _get_batch_indices(self, batch_idx, total_classes):
        """Helper method to get the start and end indices for a batch.

        Args:
            batch_idx (int): The index of the batch.
            total_classes (int): Total number of classes.

        Returns:
            (int, int): Start and end indices for the batch.
        """
        start_idx = batch_idx * self.mini_batch_size
        end_idx = min((batch_idx + 1) * self.mini_batch_size, total_classes)
        return start_idx, end_idx

    def forward(self, model: "ProtoPNet"):
        """Forward pass that calculates the Grassmannian Orthogonality Loss.

        Args:
            model (ProtoPNet): The ProtoPNet model containing prototype tensors.

        Returns:
            torch.Tensor: The computed Grassmannian Orthogonality loss.
        """
        prototype_tensors = self._reshape_and_normalize_prototypes(model)

        # Compute the projection matrices for each class
        # projection_matrices shape: (num_classes, num_vectors, num_vectors)
        projection_matrices = prototype_tensors.transpose(-2, -1) @ prototype_tensors

        # Compute loss with or without mini-batching
        if self.mini_batch_size is not None:
            grassmannian_loss = self._compute_loss_with_minibatch(
                projection_matrices, model
            )
        else:
            grassmannian_loss = self._compute_loss_without_minibatch(
                projection_matrices, model
            )

        return grassmannian_loss


class ClassificationBoundaryLoss(nn.Module):
    """
    Computes the prototype regularization loss for classification boundaries.

    This module implements the regularization terms introduced in ST ProtoPNet
    (https://arxiv.org/abs/2301.04011), which adjust prototype placement
    relative to the classification boundary using two complementary strategies:

    Minimization (Closeness Loss ℓ_cls): Promotes inter-class similarity by encouraging
    prototypes of different classes to move closer together, supporting
    better representation learning across class boundaries.

    Maximization (Discrimination Loss ℓ_dsc): Enhances inter-class separability by
    minimizing the pairwise similarities between prototypes from different classes.

    These loss components balance prototype alignment and separation
    for improved classification performance.

    Args:
        class_specific (bool): If class specific prototypes not set, no loss is generated.

    Raises:
        NotImplementedError: class children define the boundary loss calculation.
    """

    def __init__(self, class_specific=True):
        # Ensure only one of the two options is True
        if not class_specific:
            warnings.warn(
                "Classification boundary closeness was specified why without class specific setting."
                "This loss function will not contribute to loss."
            )

        super(ClassificationBoundaryLoss, self).__init__()
        self.class_specific = class_specific
        self.name = "classification_boundary_loss"

    def _inter_class_pairwise_prototype_similarity(
        self, prototypes_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the pairwise similarity between classes by calculating the
        similarity between prototypes of different classes.

        Args:
            prototypes_matrix (torch.Tensor): Prototypes organized by class.

        Returns:
            torch.Tensor : Pairwise prototype similarity between classes.
        """
        (
            num_classes,
            num_prototypes_per_class,
            latent_channels,
            height,
            width,
        ) = prototypes_matrix.shape

        # Reshape prototypes_matrix for dot product computation
        flattened_prototypes = prototypes_matrix.reshape(
            num_classes * num_prototypes_per_class, latent_channels, height, width
        )

        # Perform the dot product directly using einsum
        simi_dot = torch.einsum(
            "ijkl,mjkl->im", flattened_prototypes, flattened_prototypes
        )

        # Reshape and return to desired output shape
        simi_dot = simi_dot.reshape(
            num_classes, num_prototypes_per_class, num_classes, num_prototypes_per_class
        )
        return simi_dot.permute(0, 2, 1, 3).reshape(num_classes, num_classes, -1)

    def _boundary_loss(simi_dot: torch.Tensor, I_operator_c: torch.Tensor) -> float:
        # this is only defined by child - not sure how to make it well defined
        # for tests
        raise NotImplementedError("This method should be overridden by subclasses.")

    def forward(self, model: "ProtoPNet"):
        """
        Computes the prototype regularization loss for classification boundaries.

        Args:
            model (ProtoPNet): The ProtoPNet model containing prototype tensors.

        Returns:
            torch.Tensor: The computed classification boundary closeness loss.
        """
        device = getattr(
            model, "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        loss = 0.0

        if self.class_specific:
            # set variables for simplicity
            num_classes = model.prototype_layer.num_classes
            num_prototypes_per_class = model.prototype_layer.num_prototypes_per_class
            latent_dim, height, width = model.prototype_layer.prototype_tensors.shape[
                1:
            ]  # more elegant way to get these values?
            prototypes = model.prototype_layer.prototype_tensors

            # Shape: [num_classes * num_prototypes_per_class, latent_dim, height, width]
            prototypes_matrix = prototypes.reshape(
                num_classes,
                num_prototypes_per_class,
                latent_dim,
                height,
                width,
            )

            # Shape: [num_classes, num_classes]
            I_operator_c = 1 - torch.eye(num_classes, num_classes).to(device)

            # Shape: [num_classes, num_classes, num_prototypes_per_class^2]
            simi_dot = self._inter_class_pairwise_prototype_similarity(
                prototypes_matrix
            )

            loss += self._boundary_loss(simi_dot, I_operator_c)

            del prototypes_matrix

        return loss


class ClosenessLoss(ClassificationBoundaryLoss):
    """
    Extends the prototype regularization loss for classification boundaries.

    This module implements the following regularization term:

    Minimization (Closeness Loss ℓ_cls): Promotes inter-class similarity by encouraging
    prototypes of different classes to move closer together, supporting
    better representation learning across class boundaries.

    Args:
        class_specific (bool): If class specific prototypes not set, no loss is generated.
    """

    def __init__(self, class_specific=True):
        super().__init__(class_specific)
        self.name = "closeness_loss"

    def _boundary_loss(
        self, simi_dot: torch.Tensor, I_operator_c: torch.Tensor
    ) -> float:
        """
        Computes the closeness loss for classification boundaries.

        Args:
            simi_dot (torch.Tensor): Pairwise prototype similarity between classes.
            I_operator_c (torch.Tensor): Complement of torch.eye(.) with shape [num_classes, num_classes].

        Returns:
            float: loss calculation.
        """
        # Shape: [num_classes, num_classes]
        simi_dot_support_min = torch.min(simi_dot, dim=-1)[0]
        return (-simi_dot_support_min * I_operator_c).sum() / I_operator_c.sum()


class DiscriminationLoss(ClassificationBoundaryLoss):
    """
    Extends the prototype regularization loss for classification boundaries.

    This module implements the following regularization term:

    Maximization (Discrimination Loss ℓ_dsc): Enhances inter-class separability by
    minimizing the pairwise similarities between prototypes from different classes.

    Args:
        class_specific (bool): If class specific prototypes not set, no loss is generated.
    """

    def __init__(self, class_specific=True):
        super().__init__(class_specific)
        self.name = "discrimination_loss"

    def _boundary_loss(
        self, simi_dot: torch.Tensor, I_operator_c: torch.Tensor
    ) -> float:
        """
        Computes the discrimination loss for classification boundaries.

        Args:
            simi_dot (torch.Tensor): Pairwise prototype similarity between classes.
            I_operator_c (torch.Tensor): Complement of torch.eye(.) with shape [num_classes, num_classes].

        Returns:
            float: loss calculation.
        """
        # Shape: [num_classes, num_classes]
        simi_dot_trivial_max = torch.max(simi_dot, dim=-1)[0]
        return (simi_dot_trivial_max * I_operator_c).sum() / I_operator_c.sum()


class SerialFineAnnotationCost(nn.Module):
    def __init__(self):
        super(SerialFineAnnotationCost, self).__init__()

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        white_coef=None,
    ):
        valid_indices = (
            fine_annotation.view(fine_annotation.shape[0], -1).sum(dim=1) > 0
        )
        target = target[valid_indices]
        upsampled_activation = upsampled_activation[valid_indices]
        fine_annotation = fine_annotation[valid_indices]

        prototype_targets = prototype_class_identity.argmax(dim=1)
        v, i = prototype_targets.sort()
        if (v != prototype_targets).all():
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototypes are not grouped together."
            )
        _, class_counts = prototype_targets.unique(return_counts=True)
        unique_counts = class_counts.unique()
        if len(unique_counts) != 1:
            raise NotImplementedError(
                "Do not use Serial Fine Annotation cost when prototype classes are imbalanced."
            )

        proto_num_per_class = list(set(class_counts))[0]
        device = upsampled_activation.device

        all_white_mask = torch.ones(
            upsampled_activation.shape[2], upsampled_activation.shape[3]
        ).to(device)

        fine_annotation_cost = torch.tensor(0.0).to(device)

        for index in range(target.shape[0]):
            weight1 = 1 * all_white_mask
            weight2 = 1 * fine_annotation[index]

            if white_coef is not None:
                weight1 *= white_coef

            fine_annotation_cost += (
                torch.norm(
                    upsampled_activation[index, : target[index] * proto_num_per_class]
                    * (weight1)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        target[index]
                        * proto_num_per_class : (target[index] + 1)
                        * proto_num_per_class,
                    ]
                    * (weight2)
                )
                + torch.norm(
                    upsampled_activation[
                        index,
                        (target[index] + 1) * proto_num_per_class :,
                    ]
                    * (weight1)
                )
            )

        return fine_annotation_cost


class GenericFineAnnotationCost(nn.Module):
    def __init__(self, scoring_function):
        """
        Parameters:
        ----------
        scoring_function (function): Function for aggregating the loss costs. Will receive the masked activations.
        """
        super(GenericFineAnnotationCost, self).__init__()
        self.scoring_function = scoring_function

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        upsampled_activation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
    ):
        """
        Calculates the fine-annotation loss for a given set of inputs.

        Parameters:
        ----------
            target (torch.Tensor): Tensor of targets. Size(Batch)
            upsampled_activation (torch.Tensor): Size(batch, n_prototypes, height, width)
            fine_annotation (torch.Tensor): Fine annotation tensor Size(batch, 1, height, width)
            prototype_class_identity (torch.Tensor): Class identity tensor for prototypes size(num_prototypes, num_classes)

        Returns:
        --------
            fine_annotation_loss (torch.Tensor): Fine annotation loss tensor

        Notes:
        -----
            This function assumes that the input tensors are properly aligned such that the prototype at index i
            in the `upsampled_activation` tensor corresponds to the class at index i in the `prototype_class_identity`
            tensor.

        Called in following files:
            - train_and_eval.py: l2_fine_annotation_loss(), square_fine_annotation_loss()

        """
        target_set = target.unique()
        class_fa_losses = torch.zeros(target_set.shape[0])

        valid_indices = (
            fine_annotation.view(fine_annotation.shape[0], -1).sum(dim=1) > 0
        )
        target = target[valid_indices]
        upsampled_activation = upsampled_activation[valid_indices]
        fine_annotation = fine_annotation[valid_indices]

        # Assigned but never used in IAIA-BL
        # total_proto = upsampled_activation.shape[1]

        # unhot the one-hot encoding
        prototype_targets = prototype_class_identity.argmax(
            dim=1
        )  # shape: (n_prototype)

        # This shifts our iteration from O(n) to O(#targets)
        for target_val in list(target_set):
            # We have different calculations depending on whether or not the prototype
            # is in class or not, so we will find each group
            in_class_targets = target == target_val  # shape: (batch)
            in_class_prototypes = (
                prototype_targets == target_val
            )  # shape: (n_prototypes)

            # In Class case Size(D', p=y, 244, 244)
            prototype_activation_in_class = upsampled_activation[in_class_targets][
                :, in_class_prototypes, :, :
            ]
            # broadcast fine_annotation to prototypes in dim 1
            prototypes_activation_in_class_masked = (
                prototype_activation_in_class * fine_annotation[in_class_targets]
            )

            # Out of class case Size(batch, p!=y, 244, 244)
            prototype_activation_out_of_class = upsampled_activation[in_class_targets][
                :, ~in_class_prototypes, :, :
            ]

            # regroup after masking to parallelize, Size(batch, p, 244, 244)
            class_activations = torch.cat(
                (
                    prototypes_activation_in_class_masked,
                    prototype_activation_out_of_class,
                ),
                1,
            )

            # Size(D', p) - norms for all prototypes
            class_fa_for_all_prototypes = self.scoring_function(class_activations)

            class_fa_losses[target_val] = torch.sum(class_fa_for_all_prototypes)

        fine_annotation_loss = class_fa_losses.sum()
        return fine_annotation_loss


class FineAnnotationCost(nn.Module):
    def __init__(self, fa_loss: str = "serial"):
        super(FineAnnotationCost, self).__init__()

        self.fa_loss = fa_loss
        self.name = "fine_annotation"
        self.required_forward_results = {
            "target",
            "fine_annotation",
            "prototype_class_identity",
            "upsampled_activation",
        }

        # TODO: Could choose just one cost function here
        # And then determine necessary parameters as kwdict
        # Make it more generic
        self.serial_cost = SerialFineAnnotationCost()
        self.l2_fine_annotation_loss = GenericFineAnnotationCost(self.l2_scoring)
        self.square_fine_annotation_loss = GenericFineAnnotationCost(
            self.square_scoring
        )

        assert self.fa_loss in ["serial", "l2_norm", "square"]

    def forward(
        self,
        target: torch.Tensor,
        fine_annotation: torch.Tensor,
        prototype_class_identity: torch.Tensor,
        upsampled_activation: torch.Tensor,
    ):
        target = torch.tensor(target).int()
        if fine_annotation is None:
            fa_shape = list(upsampled_activation.shape)
            fa_shape[1] = 1
            fine_annotation = torch.zeros(fa_shape).to(upsampled_activation.device)
        elif fine_annotation.is_sparse:
            fine_annotation = fine_annotation.to_dense()

        if self.fa_loss == "serial":
            fine_annotation_cost = self.serial_cost(
                target, fine_annotation, upsampled_activation, prototype_class_identity
            )
        elif self.fa_loss == "l2_norm":
            fine_annotation_cost = self.l2_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )
        elif self.fa_loss == "square":
            fine_annotation_cost = self.square_fine_annotation_loss(
                target,
                fine_annotation,
                upsampled_activation,
                prototype_class_identity,
            )

        return fine_annotation_cost

    def l2_scoring(self, activations):
        return activations.norm(p=2, dim=(2, 3))

    def square_scoring(self, activations):
        return activations.square().sum(dim=(2, 3))


class ContrastiveMaskedPatchSimilarity(nn.Module):
    def __init__(self, activation: nn.Module, masked: bool = True):
        """
        Args:
            activation (nn.Module): Activation function to apply to the masked and unmasked latent tensors.
            masked (bool): Whether to calculate the the similarity between masked (True) or unmasked (False) areas. Mask values are 1 for true and 0 for false.
        """
        super(ContrastiveMaskedPatchSimilarity, self).__init__()
        self.name = f"contrastive_{'masked' if masked else 'unmasked'}"
        self.activation = activation
        self.masked = masked

        self.required_forward_results = {
            "unmasked_latent_tensors",
            "masked_latent_tensors",
            "latent_mask",
        }

    def forward(
        self,
        unmasked_latent_tensors: torch.Tensor,
        masked_latent_tensors: torch.Tensor,
        latent_mask: torch.Tensor,
    ):
        """
        Calculate the similarity between the masked and unmasked latent tensors in areas where the mask is not applied.
        Args:
            unmasked_latent_tensors (torch.Tensor): The latent tensors without the mask. Size(batch, channel, latent_height, latent_width)
            masked_latent_tensors (torch.Tensor): The latent tensors with the mask. Size(batch, channel, latent_height, latent_width)
            latent_mask (torch.Tensor): The mask tensor. Size(batch, 1, latent_height, latent_width)
        """
        assert (
            len(unmasked_latent_tensors.shape) == 4
            and len(masked_latent_tensors.shape) == 4
        ), "tensors must be B x C x H x W"
        assert len(latent_mask.shape) == 3, "mask must be B x H x W"

        batch_size, channels = (
            unmasked_latent_tensors.shape[0],
            unmasked_latent_tensors.shape[1],
        )

        pseudo_proto_slots = (
            masked_latent_tensors.shape[2] * masked_latent_tensors.shape[3]
        )

        prototype_like_masked = masked_latent_tensors.permute(0, 2, 3, 1).reshape(
            (batch_size * pseudo_proto_slots, channels, 1, 1)
        )

        fully_cross_activation = self.activation(
            unmasked_latent_tensors, prototype_like_masked
        )

        unmasked_idxs = torch.arange(batch_size).repeat_interleave(pseudo_proto_slots)
        pseudo_proto_idxs = torch.arange(batch_size * pseudo_proto_slots)

        pairwise_activation = fully_cross_activation[
            unmasked_idxs, pseudo_proto_idxs, :, :
        ]

        selected_pairwise_activations = (
            pairwise_activation.reshape(
                (batch_size, pseudo_proto_slots, pseudo_proto_slots)
            )
            .diagonal(dim1=1, dim2=2)
            .reshape(
                (
                    batch_size,
                    masked_latent_tensors.shape[2],
                    masked_latent_tensors.shape[3],
                )
            )
        )

        if not self.masked:
            latent_mask = (latent_mask == 0).to(dtype=torch.int8)

        masked_pairwise_activations = selected_pairwise_activations[
            latent_mask.nonzero(as_tuple=True)
        ]

        return masked_pairwise_activations.mean()
