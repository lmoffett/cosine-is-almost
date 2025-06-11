import pytest
import torch
import torch.nn as nn

from protopnet.losses import (
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
from protopnet.model_losses import ProtoPNetLoss, loss_for_coefficients


def test_cross_entropy_only():
    coefs = {"cross_entropy": 1.0}
    loss = loss_for_coefficients(coefs)

    # assert isinstance(loss, ProtoPNetLoss)
    assert len(loss.batch_loss.batch_losses) == 1
    assert isinstance(loss.batch_loss.batch_losses[0].loss, CrossEntropyCost)
    assert loss.batch_loss.batch_losses[0].coefficient == 1.0
    assert len(loss.model_regularization.model_losses) == 0


def test_all_terms_default_flags():
    coefs = {
        "cross_entropy": 1.0,
        "nll_loss": 0.5,
        "cluster": 0.7,
        "separation": 0.6,
        "l1": 0.2,
        "orthogonality_loss": 0.3,
        "grassmannian_orthogonality_loss": 0.4,
        "closeness_loss": 0.8,
        "discrimination_loss": 0.9,
    }
    loss = loss_for_coefficients(coefs)

    # assert isinstance(loss, ProtoPNetLoss)
    assert len(loss.batch_loss.batch_losses) == 4
    assert len(loss.model_regularization.model_losses) == 5

    # Assertions for batch losses
    assert isinstance(loss.batch_loss.batch_losses[0].loss, CrossEntropyCost)
    assert loss.batch_loss.batch_losses[0].coefficient == 1.0

    assert isinstance(loss.batch_loss.batch_losses[1].loss, NegativeLogLikelihoodCost)
    assert loss.batch_loss.batch_losses[1].coefficient == 0.5

    assert isinstance(loss.batch_loss.batch_losses[2].loss, ClusterCost)
    assert loss.batch_loss.batch_losses[2].coefficient == 0.7
    assert loss.batch_loss.batch_losses[2].loss.class_specific is True  # Default value

    assert isinstance(loss.batch_loss.batch_losses[3].loss, SeparationCost)
    assert loss.batch_loss.batch_losses[3].coefficient == 0.6

    # Assertions for model regularization losses
    assert isinstance(
        loss.model_regularization.model_losses[0].loss, L1CostClassConnectionLayer
    )
    assert loss.model_regularization.model_losses[0].coefficient == 0.2

    assert isinstance(loss.model_regularization.model_losses[1].loss, OrthogonalityLoss)
    assert loss.model_regularization.model_losses[1].coefficient == 0.3
    assert loss.model_regularization.model_losses[1].loss.p == 2  # Default value

    assert isinstance(
        loss.model_regularization.model_losses[2].loss, GrassmannianOrthogonalityLoss
    )
    assert loss.model_regularization.model_losses[2].coefficient == 0.4
    assert (
        loss.model_regularization.model_losses[2].loss.normalize is True
    )  # Default value
    assert (
        loss.model_regularization.model_losses[2].loss.mini_batch_size is None
    )  # Default value

    assert isinstance(loss.model_regularization.model_losses[3].loss, ClosenessLoss)
    assert loss.model_regularization.model_losses[3].coefficient == 0.8
    assert (
        loss.model_regularization.model_losses[3].loss.class_specific is True
    )  # Default value

    assert isinstance(
        loss.model_regularization.model_losses[4].loss, DiscriminationLoss
    )
    assert loss.model_regularization.model_losses[4].coefficient == 0.9
    assert (
        loss.model_regularization.model_losses[4].loss.class_specific is True
    )  # Default value


def test_all_terms_with_flags():
    coefs = {
        "cross_entropy": 1.0,
        "nll_loss": 0.5,
        "cluster": 0.7,
        "separation": 0.6,
        "l1": 0.2,
        "orthogonality_loss": 0.3,
        "grassmannian_orthogonality_loss": 0.4,
        "closeness_loss": 0.8,
        "discrimination_loss": 0.9,
        "fa": 1.0,
    }
    loss = loss_for_coefficients(
        coefs,
        class_specific_cluster=False,
        ortho_p_norm=1,
        grassmannian_orthogonality_loss_normalize=False,
        grassmannian_orthogonality_loss_mini_batch_size=32,
        fa_func="l2_norm",
    )

    # assert isinstance(loss, ProtoPNetLoss)
    assert len(loss.batch_loss.batch_losses) == 5
    assert len(loss.model_regularization.model_losses) == 5

    # Assertions for batch losses
    assert isinstance(loss.batch_loss.batch_losses[0].loss, CrossEntropyCost)
    assert loss.batch_loss.batch_losses[0].coefficient == 1.0

    assert isinstance(loss.batch_loss.batch_losses[1].loss, NegativeLogLikelihoodCost)
    assert loss.batch_loss.batch_losses[1].coefficient == 0.5

    assert isinstance(loss.batch_loss.batch_losses[2].loss, ClusterCost)
    assert loss.batch_loss.batch_losses[2].coefficient == 0.7
    assert loss.batch_loss.batch_losses[2].loss.class_specific is False  # Updated flag

    assert isinstance(loss.batch_loss.batch_losses[3].loss, SeparationCost)
    assert loss.batch_loss.batch_losses[3].coefficient == 0.6

    assert isinstance(loss.batch_loss.batch_losses[4].loss, FineAnnotationCost)
    assert loss.batch_loss.batch_losses[4].coefficient == 1.0
    assert loss.batch_loss.batch_losses[4].loss.name == "fine_annotation"

    # Assertions for model regularization losses
    assert isinstance(
        loss.model_regularization.model_losses[0].loss, L1CostClassConnectionLayer
    )
    assert loss.model_regularization.model_losses[0].coefficient == 0.2

    assert isinstance(loss.model_regularization.model_losses[1].loss, OrthogonalityLoss)
    assert loss.model_regularization.model_losses[1].coefficient == 0.3
    assert loss.model_regularization.model_losses[1].loss.p == 1  # Updated flag

    assert isinstance(
        loss.model_regularization.model_losses[2].loss, GrassmannianOrthogonalityLoss
    )
    assert loss.model_regularization.model_losses[2].coefficient == 0.4
    assert (
        loss.model_regularization.model_losses[2].loss.normalize is False
    )  # Updated flag
    assert (
        loss.model_regularization.model_losses[2].loss.mini_batch_size == 32
    )  # Updated flag

    assert isinstance(loss.model_regularization.model_losses[3].loss, ClosenessLoss)
    assert loss.model_regularization.model_losses[3].coefficient == 0.8
    assert (
        loss.model_regularization.model_losses[3].loss.class_specific is False
    )  # Updated flag

    assert isinstance(
        loss.model_regularization.model_losses[4].loss, DiscriminationLoss
    )
    assert loss.model_regularization.model_losses[4].coefficient == 0.9
    assert (
        loss.model_regularization.model_losses[4].loss.class_specific is False
    )  # Updated flag


def test_ProtoPNetLoss_forward():
    logits = torch.tensor(
        [[1000000.0, 0.0, 0.0], [0.0, 1000000.0, 0.0], [0.0, 0.0, 1000000.0]]
    )
    target = torch.tensor([0, 1, 2])

    class MockProtoPNet(nn.Module):
        def __init__(self, weights):
            super(MockProtoPNet, self).__init__()
            self.prototype_prediction_head = nn.Module()
            self.prototype_prediction_head.class_connection_layer = nn.Module()
            self.prototype_prediction_head.class_connection_layer.weight = nn.Parameter(
                weights
            )

    # this is the expected loss without the coef multiplier
    model = MockProtoPNet(torch.tensor([[1.0]]))

    coefs = {
        "cross_entropy": 1.0,
        "l1": 2.0,
    }

    batch_losses = [
        LossTerm(loss=CrossEntropyCost(), coefficient=coefs["cross_entropy"])
    ]
    model_losses = [
        LossTerm(
            loss=L1CostClassConnectionLayer(negative_classes_only=False),
            coefficient=coefs["l1"],
        )
    ]

    protopnetloss = ProtoPNetLoss(batch_losses=batch_losses, model_losses=model_losses)

    loss, loss_term_dict = protopnetloss(target, model, logits=logits)

    expected_loss = 2.0
    assert torch.isclose(
        loss, torch.tensor(expected_loss)
    ), f"Expected {expected_loss}, got {loss}"

    assert (
        loss_term_dict["cross_entropy"] == 0
    ), f'Expected {0} for unweighted, got {loss_term_dict["cross_entropy"]}.'
    assert (
        loss_term_dict["cross_entropy_weighted"] == 0
    ), f'Expected {0} for unweighted, got {loss_term_dict["cross_entropy_weighted"]}.'
    assert (
        loss_term_dict["cross_entropy_coef"] == 1.0
    ), f'Expected {0} for unweighted, got {loss_term_dict["cross_entropy_coef"]}.'

    assert (
        loss_term_dict["l1"] == 1.0
    ), f'Expected {0} for unweighted, got {loss_term_dict["l1"]}.'
    assert (
        loss_term_dict["l1_weighted"] == 2.0
    ), f'Expected {0} for unweighted, got {loss_term_dict["l1_weighted"]}.'
    assert (
        loss_term_dict["l1_coef"] == 2.0
    ), f'Expected {0} for unweighted, got {loss_term_dict["l1_coef"]}.'
