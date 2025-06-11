import math
from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import one_hot

from protopnet.activations import CosPrototypeActivation
from protopnet.backbones import construct_backbone
from protopnet.losses import (
    AverageSeparationCost,
    ClassAwareExtraCalculations,
    ClosenessLoss,
    ClusterCost,
    ContrastiveMaskedPatchSimilarity,
    CrossEntropyCost,
    DiscriminationLoss,
    FineAnnotationCost,
    GrassmannianOrthogonalityLoss,
    L1CostClassConnectionLayer,
    OrthogonalityLoss,
    SeparationCost,
)
from protopnet.prediction_heads import LinearClassPrototypePredictionHead
from protopnet.prototype_layers import ClassAwarePrototypeLayer, PrototypeLayer
from protopnet.prototypical_part_model import ProtoPNet


def test_cross_entropy_cost():
    # Do not need to test hard as it is a simple wrapper around torch.nn.functional.cross_entropy

    cost = CrossEntropyCost()

    logits = torch.tensor(
        [[1000000.0, 0.0, 0.0], [0.0, 1000000.0, 0.0], [0.0, 0.0, 1000000.0]]
    )
    targets = torch.tensor([0, 1, 2])

    # Cost expects the logits as opposed to probabilities
    loss = cost(logits, targets)
    assert loss == 0, loss


# Expanded parameterized test for class-specific cluster cost calculation
@pytest.mark.parametrize(
    "similarity_score_to_each_prototype, prototypes_of_correct_class, expected_cost",
    [
        # 1D Tensor
        # pytest.param(
        #     torch.tensor([0.1, 0.2, 0.3]),
        #     torch.tensor([0, 1, 0], dtype=torch.float32),
        #     torch.mean(torch.tensor([0.2])),
        #     id="1d_tensor",
        # ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
            torch.mean(torch.tensor([0.2, 0.4, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.float32),
            torch.mean(torch.tensor([0.2, 0.3, 0.5, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.tensor(
                [[[[0], [1]], [[1], [0]]], [[[1], [0]], [[0], [1]]]],
                dtype=torch.float32,
            ),
            torch.mean(torch.tensor([0, 0, 0.5, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_cluster_cost_class_specific(
    similarity_score_to_each_prototype,
    prototypes_of_correct_class,
    expected_cost,
):
    cluster_cost = ClusterCost(class_specific=True)
    loss = cluster_cost(similarity_score_to_each_prototype, prototypes_of_correct_class)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


def test_cluster_cost_1d():
    cluster_cost = ClusterCost()
    similarity_score_to_each_prototype = torch.tensor([0.1, 0.2, 0.3])
    with pytest.raises(AssertionError):
        cluster_cost(similarity_score_to_each_prototype)


# Expanded parameterized test for non-class-specific cluster cost calculation
@pytest.mark.parametrize(
    "similarity_score_to_each_prototype, expected_cost",
    [
        # 1D Tensor
        # pytest.param(
        #     torch.tensor([0.1, 0.2, 0.3]),
        #     torch.mean(torch.tensor([0.3])),
        #     id="1d_tensor",
        # ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.mean(torch.tensor([0.3, 0.6, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.mean(torch.tensor([0.3, 0.4, 0.7, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.mean(torch.tensor([0.1, 0.4, 0.5, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_cluster_cost_non_class_specific(
    similarity_score_to_each_prototype, expected_cost
):
    cluster_cost = ClusterCost(class_specific=False)
    loss = cluster_cost(similarity_score_to_each_prototype)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


@pytest.mark.parametrize(
    "incorrect_class_prototype_activations, expected_cost",
    [
        # 1D Tensor
        pytest.param(
            torch.tensor([0.1, 0.2, 0.3]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3])),
            id="1d_tensor",
        ),
        # 2D Tensor
        pytest.param(
            torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])),
            id="2d_tensor",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            torch.mean(torch.tensor([0.0, 0.0, 0.0])),
            id="2d_all_zeros",
        ),
        # 3D Tensor
        pytest.param(
            torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]),
            torch.mean(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
            id="3d_tensor",
        ),
        # 4D Tensor with negative values
        pytest.param(
            torch.tensor(
                [[[[0.1], [-0.2]], [[-0.3], [0.4]]], [[[0.5], [-0.6]], [[-0.7], [0.8]]]]
            ),
            torch.mean(torch.tensor([0.1, -0.2, -0.3, 0.4, 0.5, -0.6, -0.7, 0.8])),
            id="4d_negative_values",
        ),
    ],
)
def test_separation_cost(incorrect_class_prototype_activations, expected_cost):
    separation_cost = SeparationCost()
    loss = separation_cost(incorrect_class_prototype_activations)
    assert torch.isclose(loss, expected_cost), f"Expected {expected_cost}, got {loss}"


def test_separation_cost_with_none():
    separation_cost = SeparationCost()
    with pytest.raises(ValueError):
        separation_cost(None)


@pytest.mark.parametrize(
    "weights, expected_l1_cost",
    [
        # Single weight
        pytest.param(torch.tensor([[1.0]]), 1.0, id="single_weight"),
        # 2x2 Matrix with positive and negative values
        pytest.param(torch.tensor([[1.0, -2.0], [3.0, -4.0]]), 10.0, id="2x2_matrix"),
        # 2x2 Matrix with all zeros
        pytest.param(torch.zeros((2, 2)), 0.0, id="2x2_zeros"),
        # 3x3 Matrix with mixed values
        pytest.param(
            torch.tensor([[1.0, -1.0, 2.0], [3.0, -3.0, 4.0], [5.0, -5.0, 6.0]]),
            30.0,
            id="3x3_mixed_values",
        ),
        # 4x4 Matrix with random values (expected cost to be calculated)
        pytest.param(torch.randn((4, 4)), "dynamic", id="4x4_random"),
    ],
)
def test_l1_cost_class_connection_layer_for_all(weights, expected_l1_cost):
    class MockProtoPNet(nn.Module):
        def __init__(self, weights):
            super(MockProtoPNet, self).__init__()
            self.prototype_prediction_head = nn.Module()
            self.prototype_prediction_head.class_connection_layer = nn.Module()
            self.prototype_prediction_head.class_connection_layer.weight = nn.Parameter(
                weights
            )

    model = MockProtoPNet(weights)
    # Check loss for class-specific L1 cost based on negative_classes_only **kwargs
    l1_cost_layer = L1CostClassConnectionLayer(negative_classes_only=False)
    l1_cost = l1_cost_layer(model)

    if expected_l1_cost == "dynamic":
        expected_l1_cost = weights.abs().sum().item()

    assert torch.isclose(
        l1_cost, torch.tensor(expected_l1_cost)
    ), f"Expected {expected_l1_cost}, got {l1_cost.item()}"


@pytest.mark.parametrize(
    "weights, expected_l1_cost",
    [
        # 2x2 Matrix with positive and negative values
        pytest.param(torch.tensor([[1.0, -2.0], [3.0, -4.0]]), 5.0, id="2x2_matrix"),
        # 2x2 Matrix with all zeros
        pytest.param(torch.zeros((2, 2)), 0.0, id="2x2_zeros"),
        # 3x3 Matrix with mixed values
        pytest.param(
            torch.tensor([[1.0, -1.0, 2.0], [3.0, -3.0, 4.0], [5.0, -5.0, 6.0]]),
            20.0,
            id="3x3_mixed_values",
        ),
        # 4x4 Matrix with random values (expected cost to be calculated)
        pytest.param(torch.randn((4, 4)), "dynamic", id="4x4_random"),
    ],
)
def test_l1_cost_class_connection_layer_for_negative_only(weights, expected_l1_cost):
    class_identity = torch.diag(torch.ones(weights.shape[0]))

    class MockProtoPNet(nn.Module):
        def __init__(self, weights):
            super(MockProtoPNet, self).__init__()
            self.prototype_prediction_head = nn.Module()
            self.prototype_prediction_head.prototype_class_identity = class_identity
            self.prototype_prediction_head.class_connection_layer = nn.Module()
            self.prototype_prediction_head.class_connection_layer.weight = nn.Parameter(
                weights
            )

    model = MockProtoPNet(weights)
    # Check loss for class-specific L1 cost based on negative_classes_only **kwargs
    l1_cost_layer = L1CostClassConnectionLayer(negative_classes_only=True)
    l1_cost = l1_cost_layer(model)

    if expected_l1_cost == "dynamic":
        expected_l1_cost = (weights - weights * class_identity).abs().sum().item()

    assert torch.isclose(
        l1_cost, torch.tensor(expected_l1_cost)
    ), f"Expected {expected_l1_cost}, got {l1_cost.item()}"


@pytest.mark.parametrize(
    "incorrect_class_prototype_activations, prototypes_of_wrong_class, expected_cost",
    [
        # Basic test with uniform wrong class prototypes
        pytest.param(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0], [1.0], [1.0]]),
            2.0,
            id="uniform_wrong_class_prototypes",
        ),
        # Test with varying wrong class prototypes
        pytest.param(
            torch.tensor([10.0, 20.0, 30.0]),
            torch.tensor([[2.0], [4.0], [6.0]]),
            5.0,
            id="varying_wrong_class_prototypes",
        ),
        # Test with zeros in activations, should handle division by zero
        pytest.param(
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([[1.0], [2.0], [3.0]]),
            0.0,
            id="zero_activations",
        ),
        # Test with all zeros in wrong class prototypes
        pytest.param(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[0.0], [0.0], [0.0]]),
            float(
                "inf"
            ),  # Expecting infinity or a very large number due to division by zero
            id="zero_wrong_class_prototypes",
        ),
        # 2D Tensor with uniform wrong class prototypes
        pytest.param(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[1.0], [1.0]]),
            2.5,
            id="2d_uniform_wrong_class_prototypes",
        ),
        # 3D Tensor with varying wrong class prototypes
        pytest.param(
            torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]]),
            torch.tensor([[2.0], [4.0]]),
            torch.mean(torch.tensor([5, 2.5, 10, 5, 15, 7.5, 20, 10])),
            id="3d_varying_wrong_class_prototypes",
        ),
    ],
)
def test_average_separation_cost(
    incorrect_class_prototype_activations, prototypes_of_wrong_class, expected_cost
):
    average_separation_cost = AverageSeparationCost()
    cost = average_separation_cost(
        incorrect_class_prototype_activations, prototypes_of_wrong_class
    )

    # Handling the infinity case separately as `torch.isclose` does not handle infinities.
    if expected_cost == float("inf"):
        assert cost.item() == float(
            "inf"
        ), f"Expected {expected_cost}, got {cost.item()}"
    else:
        assert torch.isclose(
            cost, torch.tensor(expected_cost)
        ), f"Expected {expected_cost}, got {cost.item()}"


def test_avg_separation_cost_1d():
    average_separation_cost = AverageSeparationCost()
    similarity_score_to_each_prototype = torch.tensor([0.1, 0.2, 0.3])
    prototypes_of_wrong_class = torch.tensor([0.1, 0.2, 0.3])

    with pytest.raises(AssertionError):
        average_separation_cost(
            similarity_score_to_each_prototype, prototypes_of_wrong_class
        )


def test_orthogonality_loss_0_for_ortho():
    """
    Evaluates whether, given orthogonal prototypes,
    our orthogonality loss is 0
    """
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_shape = (num_classes * 2, 512, 1, 1)

    proto_activation = CosPrototypeActivation()

    prototype_layer = ClassAwarePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        activation_function=proto_activation,
        prototype_dimension=(1, 1),
    )

    proto_vals = torch.zeros(prototype_shape)
    for k in range(prototype_shape[0]):
        proto_vals[k, k] = 1

    prototype_layer.prototype_tensors = torch.nn.Parameter(proto_vals)

    prediction_head = LinearClassPrototypePredictionHead(
        prototype_class_identity=prototype_class_identity
    )

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )
    orthogonality_loss = OrthogonalityLoss()

    assert torch.isclose(orthogonality_loss(protopnet), torch.tensor(0.0))


def test_orthogonality_loss_1_for_colin():
    """
    Evaluates whether, given orthogonal prototypes,
    our orthogonality loss is 0
    """
    backbone = construct_backbone("resnet18")

    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    prototype_shape = (num_classes * 2, 512, 3, 3)

    proto_activation = CosPrototypeActivation()

    prototype_layer = ClassAwarePrototypeLayer(
        prototype_class_identity=prototype_class_identity,
        activation_function=proto_activation,
        prototype_dimension=(3, 3),
    )

    proto_vals = torch.zeros(prototype_shape)
    for k in range(prototype_shape[0]):
        proto_vals[k, 0] = 1

    prototype_layer.prototype_tensors = torch.nn.Parameter(proto_vals)

    prediction_head = LinearClassPrototypePredictionHead(
        prototype_class_identity=prototype_class_identity
    )

    protopnet = ProtoPNet(
        backbone,
        torch.nn.Identity(),
        proto_activation,
        prototype_layer,
        prediction_head,
        warn_on_errors=True,
    )
    orthogonality_loss = OrthogonalityLoss()

    # We will have a n_proto_parts_per_class x n_proto_parts_per_class matrix
    n_proto_parts_per_class = (
        (prototype_shape[0] // num_classes) * prototype_shape[-1] * prototype_shape[-2]
    )

    target_per_class = torch.norm(
        torch.ones((n_proto_parts_per_class, n_proto_parts_per_class))
        - torch.eye(n_proto_parts_per_class)
    )
    target = target_per_class * num_classes

    assert torch.isclose(orthogonality_loss(protopnet), target)


@pytest.mark.parametrize(
    "type, target, upsampled_activation, fine_annotation, prototype_class_identity, expected_cost",
    [
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "serial",
            [0, 1, 2],  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(0.0),
            id="annotation_on_everything_serial",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "serial",
            [0, 1, 2],  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(
                sum(
                    [
                        math.sqrt(100) + math.sqrt(50),
                        math.sqrt(50) + math.sqrt(50) + math.sqrt(50),
                        math.sqrt(100) + math.sqrt(50),
                    ]
                )
            ),
            id="annotation_on_nothing_serial",
        ),
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "l2_norm",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(0.0),
            id="annotation_on_everything_l2",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "l2_norm",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(90.0),
            id="annotation_on_nothing_l2",
        ),
        # Test no fine annotation (fa is all 0's) ie allowed to activate anywhere
        pytest.param(
            "square",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.zeros([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(0.0),
            id="annotation_on_everything_square",
        ),
        # Test with white-out fine annotation (fa is all 1's) ie don't want it to active anywhere
        pytest.param(
            "square",
            torch.tensor([0, 1, 2]),  # target class id
            torch.ones([3, 6, 5, 5]),  # upsampled annotation
            torch.ones([3, 1, 5, 5]),  # fine annotation
            one_hot(torch.tensor([0, 0, 1, 1, 2, 2])),  # prototype_class_identity
            torch.tensor(450.0),
            id="annotation_on_nothing_square",
        ),
    ],
)
def test_fine_annotation_cost(
    type,
    target,
    upsampled_activation,
    fine_annotation,
    prototype_class_identity,
    expected_cost,
):
    fa_cost = FineAnnotationCost(fa_loss=type)
    cost = fa_cost(
        target=target,
        fine_annotation=fine_annotation,
        prototype_class_identity=prototype_class_identity,
        upsampled_activation=upsampled_activation,
    )
    print(cost)
    assert torch.isclose(cost, expected_cost), f"Expected {expected_cost}, got {cost}"


class MockPrototypeLayer:
    def __init__(self, prototype_tensors, num_classes, num_prototypes_per_class):
        self.prototype_tensors = prototype_tensors
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.latent_channels = prototype_tensors.shape[2]


class MockProtoPNet:
    def __init__(self, prototype_tensors, num_classes, num_prototypes_per_class):
        self.prototype_layer = MockPrototypeLayer(
            prototype_tensors, num_classes, num_prototypes_per_class
        )
        self.device = prototype_tensors.device


@pytest.fixture
def grassman_loss_fn():
    return GrassmannianOrthogonalityLoss()


@pytest.fixture(params=["grassman"], scope="function")
def grassmannian_loss_fn(request, grassman_loss_fn):
    # request.param is the class to be instantiated
    if request.param == "grassman":
        return grassman_loss_fn
    else:
        raise NotImplementedError(f"Unknown loss function {request.param}")


def gram_schmidt(vectors):
    """Applies Gram-Schmidt process to orthogonalize given vectors."""
    orthogonal_vectors = []
    for v in vectors:
        for u in orthogonal_vectors:
            v = v - torch.dot(v, u) * u
        orthogonal_vectors.append(F.normalize(v, dim=0))
    return torch.stack(orthogonal_vectors)


def initialize_prototypes(num_classes, num_prototypes_per_class, channels):
    """Initializes prototype vectors for num_classes classes with num_prototypes_per_class prototypes per class"""
    prototype_vectors = []
    for _ in range(num_classes):
        random_vectors = F.normalize(
            torch.randn(num_prototypes_per_class, channels), p=2, dim=1
        )
        orthogonal_vectors = gram_schmidt(random_vectors)
        # Reshape to (num_prototypes_per_class, channels, 1, 1)
        orthogonal_vectors = orthogonal_vectors.view(
            num_prototypes_per_class, channels, 1, 1
        )
        prototype_vectors.append(orthogonal_vectors)
    return torch.cat(prototype_vectors, dim=0)


# Test for identical bases (loss should be zero)
@pytest.mark.parametrize("initializations", range(5))
def test_identical_bases_loss_zero(grassmannian_loss_fn, initializations):
    num_classes = 2
    num_prototypes_per_class = 3
    channels = 5

    # Create identical prototype vectors and normalize
    prototype_vectors = initialize_prototypes(1, num_prototypes_per_class, channels)

    # copy the prototype vector to create a second class
    prototype_vectors = torch.cat([prototype_vectors, prototype_vectors], dim=0)

    # Mock model
    model = MockProtoPNet(prototype_vectors, num_classes, num_prototypes_per_class)

    # Calculate the loss
    loss_value = grassmannian_loss_fn(model)

    # Assert loss is zero
    assert torch.isclose(
        loss_value, torch.tensor(0.0), atol=1e-05
    ), f"Expected 0.0, got {loss_value.item()}"


# Test for different bases that span the same subspace (loss should be zero)
@pytest.mark.parametrize("permutation_num", range(5))
def test_different_classes_same_subspace(grassmannian_loss_fn, permutation_num):
    """Tests if two bases from two different classes that span the same subspace have zero Grassmannian distance."""
    num_prototypes_per_class = 3
    channels = 5

    # Generate a set of random vectors and orthogonalize them using Gram-Schmidt
    original_vectors = torch.randn(num_prototypes_per_class, channels)
    basis_A = gram_schmidt(original_vectors)

    # Create a row-wise permutation of basis_A to create basis_B
    permutation = torch.randperm(num_prototypes_per_class)
    basis_B = basis_A[permutation]

    # Reshape to (num_prototypes_per_class, channels, 1, 1) as required by GrassmannianOrthogonalityLoss
    basis_A = basis_A.view(num_prototypes_per_class, channels, 1, 1)
    basis_B = basis_B.view(num_prototypes_per_class, channels, 1, 1)

    # Concatenate basis_A and basis_B to simulate prototypes for two different classes spanning the same subspace
    prototype_tensors = torch.cat([basis_A, basis_B], dim=0)

    # Create mock model with two different classes, each having prototypes that span the same subspace
    num_classes = 2
    model = MockProtoPNet(
        prototype_tensors,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    # Calculate the loss
    loss_value = grassmannian_loss_fn(model)

    # Assert loss is zero since both classes' prototypes span the same subspace
    assert torch.isclose(
        loss_value, torch.tensor(0.0), atol=1e-5
    ), f"Expected loss to be 0.0 for bases that span the same subspace, got {loss_value.item()} for basis A:\n{basis_A}\nbasis B:\n{basis_B}"


def test_non_zero_loss_for_integral_bases_r2(grassmannian_loss_fn):
    """Test hand-crafted R^2 bases to ensure non-zero Grassmannian loss."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 4

    # Spanning the R^2 subspace
    prototype_vectors_r2_unimodular = (
        torch.tensor(
            [
                # Class 1
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [-1, 1, 0, 0],
                # Class 2
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                # Class 3
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ]
        )
        .float()
        .view(num_classes * num_prototypes_per_class, channels, 1, 1)
    )

    # Create mock model for the R^2 subspace set
    model = MockProtoPNet(
        prototype_vectors_r2_unimodular,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    # Calculate the loss using the provided Grassmannian loss function
    loss_value = grassmannian_loss_fn(model)

    # Assert that the loss is non-zero since the prototypes from each class span different subspaces
    assert (
        loss_value > 0
    ), f"Expected loss to be non-zero for different subspaces in R^2, but got {loss_value.item()}"


def test_non_zero_loss_for_integral_bases_r3(grassmannian_loss_fn):
    """Test hand-crafted R^3 bases to ensure non-zero Grassmannian loss."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 4

    # Spanning the R^3 subspace
    prototype_vectors_r3_integral = (
        torch.tensor(
            [
                # Class 1
                [5, 0, 0, 0],
                [0, -7, 0, 0],
                [0, 0, 2, 0],
                # Class 2
                [0, 0, 0, 3],
                [15, 0, 0, 0],
                [0, 0, 0, -1],
                # Class 3
                [1, 0, 0, 0],
                [0, 6, 0, 0],
                [0, 0, -2, 0],
            ]
        )
        .float()
        .view(num_classes * num_prototypes_per_class, channels, 1, 1)
    )

    # Create mock model for the R^3 subspace set
    model = MockProtoPNet(
        prototype_vectors_r3_integral,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    # Calculate the loss using the provided Grassmannian loss function
    loss_value = grassmannian_loss_fn(model)

    # Assert that the loss is non-zero since the prototypes from each class span different subspaces
    assert (
        loss_value > 0
    ), f"Expected loss to be non-zero for different subspaces in R^3, but got {loss_value.item()}"


def test_non_zero_loss_for_integral_bases_r4(grassmannian_loss_fn):
    """Test hand-crafted R^4 bases to ensure non-zero Grassmannian loss."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 4

    # Spanning the R^4 subspace
    prototype_vectors_r4_integral = (
        torch.tensor(
            [
                # Class 1
                [7, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                # Class 2
                [1, 0, 1, 0],
                [0, -1, 0, 1],
                [0, 0, 0, 0],
                # Class 3
                [1, 0, -1, 0],
                [0, 1, 1, 0],
                [0, 0, 1, -1],
            ]
        )
        .float()
        .view(num_classes * num_prototypes_per_class, channels, 1, 1)
    )

    # Create mock model for the R^4 subspace set
    model = MockProtoPNet(
        prototype_vectors_r4_integral,
        num_classes=num_classes,
        num_prototypes_per_class=num_prototypes_per_class,
    )

    # Calculate the loss using the provided Grassmannian loss function
    loss_value = grassmannian_loss_fn(model)

    # Assert that the loss is non-zero since the prototypes from each class span different subspaces
    assert (
        loss_value > 0
    ), f"Expected loss to be non-zero for different subspaces in R^4, but got {loss_value.item()}"


def test_zero_loss_for_same_subspace_integral_bases(grassmannian_loss_fn):
    """Test three different integral bases that span the same subspace and ensure the Grassmannian loss is zero."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 5  # Adjust to span a subspace in R^5

    prototype_vectors_r3_class_1 = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    ).float()

    # Create other bases for Class 2 and Class 3 that span the same subspace as Class 1
    prototype_vectors_r3_class_2 = torch.tensor(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [-1, 0, 0, 0, 0],
        ]
    ).float()

    prototype_vectors_r3_class_3 = torch.tensor(
        [
            [0, 0, -1, 0, 0],
            [0, -1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    ).float()

    # Reshape to (num_prototypes_per_class, channels, 1, 1) as required by GrassmannianOrthogonalityLoss
    basis_A = prototype_vectors_r3_class_1.view(
        num_prototypes_per_class, channels, 1, 1
    )
    basis_B = prototype_vectors_r3_class_2.view(
        num_prototypes_per_class, channels, 1, 1
    )
    basis_C = prototype_vectors_r3_class_3.view(
        num_prototypes_per_class, channels, 1, 1
    )

    # Create separate mock models for each basis to test pairwise comparisons

    # Testing Basis A vs Basis B
    prototype_tensors_AB = torch.cat([basis_A, basis_B], dim=0)
    model_AB = MockProtoPNet(
        prototype_tensors_AB,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_AB = grassmannian_loss_fn(model_AB)
    assert torch.isclose(
        loss_value_AB, torch.tensor(0.0), atol=1e-5
    ), f"Expected loss to be 0.0 for bases A and B spanning the same subspace, got {loss_value_AB.item()}"

    # Testing Basis A vs Basis C
    prototype_tensors_AC = torch.cat([basis_A, basis_C], dim=0)
    model_AC = MockProtoPNet(
        prototype_tensors_AC,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_AC = grassmannian_loss_fn(model_AC)
    assert torch.isclose(
        loss_value_AC, torch.tensor(0.0), atol=1e-5
    ), f"Expected loss to be 0.0 for bases A and C spanning the same subspace, got {loss_value_AC.item()}"

    # Testing Basis B vs Basis C
    prototype_tensors_BC = torch.cat([basis_B, basis_C], dim=0)
    model_BC = MockProtoPNet(
        prototype_tensors_BC,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_BC = grassmannian_loss_fn(model_BC)
    assert torch.isclose(
        loss_value_BC, torch.tensor(0.0), atol=1e-5
    ), f"Expected loss to be 0.0 for bases B and C spanning the same subspace, got {loss_value_BC.item()}"


def test_non_zero_subspaces_r2_r3(grassmannian_loss_fn):
    """Test two different integral bases that span different subspaces in R^2 and R^3 and ensure non-zero Grassmannian loss."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 5  # Adjust to span a subspace in R^5

    # Create two bases that span different subspaces in R^2
    prototype_vectors_r2_class_1 = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [-1, 1, 0, 0, 0],
        ]
    ).float()

    prototype_vectors_r2_class_2 = torch.tensor(
        [
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]
    ).float()

    # Create two bases that span different subspaces in R^3
    prototype_vectors_r3_class_1 = torch.tensor(
        [
            [5, 0, 0, 0, 0],
            [0, -7, 0, 0, 0],
            [0, 0, 2, 0, 0],
        ]
    ).float()

    prototype_vectors_r3_class_2 = torch.tensor(
        [
            [0, 0, 0, 3, 0],
            [15, 0, 0, 0, 0],
            [0, 0, 0, -1, 0],
        ]
    ).float()

    # Reshape to (num_prototypes_per_class, channels, 1, 1) as required by GrassmannianOrthogonalityLoss
    basis_A_r2 = prototype_vectors_r2_class_1.view(
        num_prototypes_per_class, channels, 1, 1
    )
    basis_B_r2 = prototype_vectors_r2_class_2.view(
        num_prototypes_per_class, channels, 1, 1
    )
    basis_A_r3 = prototype_vectors_r3_class_1.view(
        num_prototypes_per_class, channels, 1, 1
    )

    # Create separate mock models for each basis to test pairwise comparisons

    # Testing R^2 Basis A vs R^2 Basis B
    prototype_tensors_AB_r2 = torch.cat([basis_A_r2, basis_B_r2], dim=0)
    model_AB_r2 = MockProtoPNet(
        prototype_tensors_AB_r2,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_AB_r2 = grassmannian_loss_fn(model_AB_r2)
    assert (
        loss_value_AB_r2 > 0
    ), f"Expected non-zero loss for bases A and B spanning different subspaces in R^2, got {loss_value_AB_r2.item()}"

    # Testing R^2 Basis A vs R^3 Basis A
    prototype_tensors_A_r2_A_r3 = torch.cat([basis_A_r2, basis_A_r3], dim=0)
    model_A_r2_A_r3 = MockProtoPNet(
        prototype_tensors_A_r2_A_r3,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_A_r2_A_r3 = grassmannian_loss_fn(model_A_r2_A_r3)
    assert (
        loss_value_A_r2_A_r3 > 0
    ), f"Expected non-zero loss for bases A in R^2 and R^3 spanning different subspaces, got {loss_value_A_r2_A_r3.item()}"

    # Testing R^2 Basis B vs R^3 Basis A
    prototype_tensors_B_r2_A_r3 = torch.cat([basis_B_r2, basis_A_r3], dim=0)
    model_B_r2_A_r3 = MockProtoPNet(
        prototype_tensors_B_r2_A_r3,
        num_classes=2,
        num_prototypes_per_class=num_prototypes_per_class,
    )
    loss_value_B_r2_A_r3 = grassmannian_loss_fn(model_B_r2_A_r3)
    assert (
        loss_value_B_r2_A_r3 > 0
    ), f"Expected non-zero loss for bases B in R^2 and A in R^3 spanning different subspaces, got {loss_value_B_r2_A_r3.item()}"


# Test for different bases (loss should be positive)
def test_nonzero_loss_for_different_bases(grassmannian_loss_fn):
    num_classes = 2
    num_prototypes_per_class = 3
    channels = 5

    # Create two different sets of prototype vectors
    prototype_vectors_1 = torch.randn(num_prototypes_per_class, channels)
    prototype_vectors_2 = torch.randn(num_prototypes_per_class, channels)

    # Normalize the vectors
    prototype_vectors_1 = F.normalize(prototype_vectors_1, p=2, dim=-1)
    prototype_vectors_2 = F.normalize(prototype_vectors_2, p=2, dim=-1)

    # Reshape to (num_prototypes_per_class, channels, 1, 1)
    prototype_vectors_1 = prototype_vectors_1.view(
        num_prototypes_per_class, channels, 1, 1
    )
    prototype_vectors_2 = prototype_vectors_2.view(
        num_prototypes_per_class, channels, 1, 1
    )

    # Stack to create different classes
    prototype_tensors = torch.cat([prototype_vectors_1, prototype_vectors_2], dim=0)

    # Mock model
    model = MockProtoPNet(prototype_tensors, num_classes, num_prototypes_per_class)

    # Calculate the loss
    loss_value = grassmannian_loss_fn(model)

    # Assert loss is positive
    assert loss_value > 0, f"Expected positive loss, got {loss_value.item()}"


# Helper function to calculate pairwise distance
def calculate_pairwise_distance(loss_fn, prototypes_A, prototypes_B):
    # Create mock model for two classes (prototypes A and B)
    prototype_tensors = torch.cat([prototypes_A, prototypes_B], dim=0)
    model = MockProtoPNet(
        prototype_tensors, num_classes=2, num_prototypes_per_class=prototypes_A.shape[0]
    )
    # Calculate loss value representing distance
    return loss_fn(model)


@pytest.fixture
def classification_boundary_closeness_loss_fn():
    return ClosenessLoss()


@pytest.fixture
def classification_boundary_discrimination_loss_fn():
    return DiscriminationLoss()


def test_classification_boundary_closeness_loss_fn_random_small_prototypes(
    classification_boundary_closeness_loss_fn,
    classification_boundary_discrimination_loss_fn,
):
    """Test both closeness and discrimination works for small number of prototypes, classes, and dimensions"""
    num_classes = 2
    num_prototypes_per_class = 3
    channels = 5

    # Generate random prototype vectors for each class
    prototype_vectors = torch.randn(
        num_classes * num_prototypes_per_class, channels
    ).view(
        num_classes * num_prototypes_per_class,
        channels,
        1,
        1,
    )

    # Mock model
    model = MockProtoPNet(prototype_vectors, num_classes, num_prototypes_per_class)

    # Calculate the losses
    loss_value_closeness = classification_boundary_closeness_loss_fn(model)
    loss_value_discrimination = classification_boundary_discrimination_loss_fn(model)

    # Assert that something happens, not null or error
    assert (
        loss_value_closeness is not None
    ), f"Expected loss to be non-zero, got {loss_value_closeness.item()}"
    assert (
        loss_value_discrimination is not None
    ), f"Expected loss to be non-zero, got {loss_value_discrimination.item()}"

    # Assert that they are not the same
    assert (
        loss_value_closeness != loss_value_discrimination
    ), f"Expected loss to be different, got {loss_value_closeness.item()}"


def test_classification_boundary_closeness_loss_fn_random_large_prototypes(
    classification_boundary_closeness_loss_fn,
    classification_boundary_discrimination_loss_fn,
):
    """Test both closeness and discrimination works for sizeable number of prototypes, classes, and dimensions"""
    num_classes = 4
    num_prototypes_per_class = 3
    channels = [1, 6]

    for channel in channels:
        # Generate random prototype vectors for each class
        prototype_vectors = torch.randn(
            num_classes * num_prototypes_per_class, channel
        ).view(
            num_classes * num_prototypes_per_class,
            channel,
            1,
            1,
        )

        # Mock model
        model = MockProtoPNet(prototype_vectors, num_classes, num_prototypes_per_class)

        # Calculate the losses
        loss_value_closeness = classification_boundary_closeness_loss_fn(model)
        loss_value_discrimination = classification_boundary_discrimination_loss_fn(
            model
        )

        # Assert that something happens, not null or error
        assert (
            loss_value_closeness is not None
        ), f"Expected loss to be non-null, got {loss_value_closeness.item()} for channel size {channel}"
        assert (
            loss_value_discrimination is not None
        ), f"Expected loss to be non-null, got {loss_value_discrimination.item()} for channel size {channel}"

        # Assert that they are not the same
        assert (
            loss_value_closeness != loss_value_discrimination
        ), f"Expected loss to be different, got {loss_value_closeness.item()}"


def test_classification_boundary_closeness_loss_fn_same_prototypes(
    classification_boundary_closeness_loss_fn,
    classification_boundary_discrimination_loss_fn,
):
    """Test both closeness and discrimination works for identical prototype sets."""
    num_classes = 2
    num_prototypes_per_class = 3
    channels = 5

    prototypes = torch.tensor(
        [
            [1.234, 0.567, 0.890, 0.123, 0.456],
            [0.789, 1.012, 0.345, 0.678, 0.901],
            [0.234, 0.567, 1.890, 0.123, 0.456],
        ],
    ).float()

    prototypes_same = torch.tensor(
        [
            [1.234, 0.567, 0.890, 0.123, 0.456],
            [1.234, 0.567, 0.890, 0.123, 0.456],
            [1.234, 0.567, 0.890, 0.123, 0.456],
        ],
    ).float()

    basis_A = prototypes.view(num_prototypes_per_class, channels, 1, 1)
    basis_B = prototypes.view(num_prototypes_per_class, channels, 1, 1)

    # Mock model
    model = MockProtoPNet(
        torch.cat([basis_A, basis_B], dim=0), num_classes, num_prototypes_per_class
    )

    basis_A_same = prototypes_same.view(num_prototypes_per_class, channels, 1, 1)
    basis_B_same = prototypes_same.view(num_prototypes_per_class, channels, 1, 1)

    # Mock model
    model_same = MockProtoPNet(
        torch.cat([basis_A_same, basis_B_same], dim=0),
        num_classes,
        num_prototypes_per_class,
    )

    max_dot_product = prototypes[-1] @ prototypes[-1]

    # Calculate the losses
    loss_value_closeness = classification_boundary_closeness_loss_fn(model)
    loss_value_discrimination = classification_boundary_discrimination_loss_fn(model)

    # Calculate the losses for the same prototypes
    loss_value_closeness_same = classification_boundary_closeness_loss_fn(model_same)
    loss_value_discrimination_same = classification_boundary_discrimination_loss_fn(
        model_same
    )

    # # Assert that the loss is close to 0 since
    # assert torch.isclose(
    #     loss_value_closeness, torch.tensor(0.0), atol=1e-5
    # ), f"Expected loss to be zero for identical prototypes, got {loss_value_closeness.item()}"

    # Assert that the loss is zero since the prototypes are identical
    assert torch.isclose(
        loss_value_discrimination, max_dot_product, atol=1e-5
    ), f"Expected loss to be zero for identical prototypes, got {loss_value_discrimination.item()}"

    # Assert that the losses are the same since prototypes are identical
    assert torch.isclose(
        loss_value_closeness_same, -loss_value_discrimination_same, atol=1e-5
    ), f"Expected closeness and discrimination losses to be the same for identical prototypes, got {loss_value_closeness_same.item()} and {loss_value_discrimination_same.item()}"


def test_classification_boundary_closeness_loss_fn_similar_prototypes(
    classification_boundary_closeness_loss_fn,
    classification_boundary_discrimination_loss_fn,
):
    """Test both closeness and discrimination works for similar prototype sets. Where one vector is the same and others are different."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 5

    prototypes_A = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
    ).float()

    prototypes_B = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
    ).float()

    prototypes_C = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    ).float()

    basis_A = prototypes_A.view(num_prototypes_per_class, channels, 1, 1)
    basis_B = prototypes_B.view(num_prototypes_per_class, channels, 1, 1)
    basis_C = prototypes_C.view(num_prototypes_per_class, channels, 1, 1)

    # Mock model
    model = MockProtoPNet(
        torch.cat([basis_A, basis_B, basis_C], dim=0),
        num_classes,
        num_prototypes_per_class,
    )

    # Calculate the losses
    loss_value_closeness = classification_boundary_closeness_loss_fn(model)
    loss_value_discrimination = classification_boundary_discrimination_loss_fn(model)

    # Assert that closeness is close to 0 since all prototypes are orthogonal
    assert torch.isclose(
        loss_value_closeness, torch.tensor(0.0), atol=1e-5
    ), f"Expected closeness loss to be zero for similar prototypes, got {loss_value_closeness.item()}"

    # Asser that discrimination is not 0 since there is at least one different vector in ever class
    assert (
        loss_value_discrimination > 0
    ), f"Expected discrimination loss to be non-zero for similar prototypes, got {loss_value_discrimination.item()}"


def test_classification_boundary_closeness_loss_fn_different_prototypes(
    classification_boundary_closeness_loss_fn,
    classification_boundary_discrimination_loss_fn,
):
    """Tests discrimination loss explicilty across three classes with different prototypes."""
    num_classes = 3
    num_prototypes_per_class = 3
    channels = 5

    # Ensure that each class has a prototype in the opposite direction of a another
    prototypes_A = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
    ).float()

    prototypes_B = torch.tensor(
        [
            [-1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
    ).float()

    prototypes_C = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1],
            [0, 0, -1, 0, 0],
        ],
    ).float()

    basis_A = prototypes_A.view(num_prototypes_per_class, channels, 1, 1)
    basis_B = prototypes_B.view(num_prototypes_per_class, channels, 1, 1)
    basis_C = prototypes_C.view(num_prototypes_per_class, channels, 1, 1)

    # Mock model
    model = MockProtoPNet(
        torch.cat([basis_A, basis_B, basis_C], dim=0),
        num_classes,
        num_prototypes_per_class,
    )

    # Calculate the losses
    loss_value_closeness = classification_boundary_closeness_loss_fn(model)
    loss_value_discrimination = classification_boundary_discrimination_loss_fn(model)

    expected_closeness_loss = 3.0 / num_classes  # normalized by number of classes
    expected_discrimination_loss = 1.0 / num_classes  # normalized by number of classes

    # Assert closeness loss is non-zero
    assert (
        loss_value_closeness > 0
    ), f"Expected closeness loss to be non-zero for different prototypes, got {loss_value_closeness.item()}"

    # Assert closeness loss matches the precomputed value
    assert torch.isclose(
        loss_value_closeness, torch.tensor(expected_closeness_loss), atol=1e-5
    ), f"Expected closeness loss to be {expected_closeness_loss}, got {loss_value_closeness.item()}"

    # Assert discrimination loss matches the precomputed value
    assert torch.isclose(
        loss_value_discrimination, torch.tensor(expected_discrimination_loss), atol=1e-5
    ), f"Expected discrimination loss to be {expected_discrimination_loss}, got {loss_value_discrimination.item()}"


@pytest.fixture
def contrastive_sim_data():
    batch_size, channels, height, width = 2, 3, 3, 3
    unmasked_latent = -torch.rand(batch_size, channels, height, width)
    masked_latent = unmasked_latent.clone()

    unmasked_latent[0, :, 1, 1] = 10
    unmasked_latent[1, :, 2, 2] = 20
    masked_latent[0, :, 1, 1] = 10
    masked_latent[1, :, 2, 2] = 20

    latent_mask = torch.zeros(batch_size, height, width)
    latent_mask[0, 1, 1] = 1
    latent_mask[1, 2, 2] = 1

    return unmasked_latent, masked_latent, latent_mask


@pytest.mark.parametrize("masked", [True, False])
def test_initialization(masked):
    activation = CosPrototypeActivation()
    model = ContrastiveMaskedPatchSimilarity(activation, masked)
    assert model.name == f"contrastive_{'masked' if masked else 'unmasked'}"


@pytest.mark.parametrize("diff_value,sim", [(None, 1.0), (10, 0.5)])
def test_sim_with_matching_values(diff_value, sim, contrastive_sim_data):
    unmasked_latent, masked_latent, latent_mask = contrastive_sim_data

    if diff_value:
        # positive value contrasted with negative
        masked_latent[:, :, 0, 0] = diff_value
        latent_mask[:, 0, 0] = 1

    activation = CosPrototypeActivation()
    model = ContrastiveMaskedPatchSimilarity(activation, masked=True)
    result = model.forward(unmasked_latent, masked_latent, latent_mask)

    assert torch.isclose(
        result, torch.tensor(sim), 1e-1
    ), "Forward calculation mismatch with expected result"


@pytest.mark.parametrize("multiplier", [-10, -1, 0, 1e-10, 10])
def test_sim_invariance_to_non_masked_value(multiplier, contrastive_sim_data):
    unmasked_latent, masked_latent, latent_mask = contrastive_sim_data

    masked_latent = masked_latent * multiplier
    masked_latent[0, :, 1, 1] = 10
    masked_latent[1, :, 2, 2] = 20

    activation = CosPrototypeActivation()
    model = ContrastiveMaskedPatchSimilarity(activation, masked=True)
    result = model.forward(unmasked_latent, masked_latent, latent_mask)

    assert torch.isclose(
        result, torch.tensor(1.0), 1e-1
    ), "Forward calculation mismatch with expected result"


@pytest.mark.parametrize("diff_value,sim", [(None, 1.0), (10, 0.9)])
def test_sim_with_matching_values(diff_value, sim, contrastive_sim_data):
    unmasked_latent, masked_latent, latent_mask = contrastive_sim_data

    if diff_value:
        # positive value contrasted with negative
        # this is already in the negative mask
        masked_latent[:, :, 0, 0] = diff_value

    activation = CosPrototypeActivation()
    model = ContrastiveMaskedPatchSimilarity(activation, masked=False)
    result = model.forward(unmasked_latent, masked_latent, latent_mask)

    assert torch.isclose(result, torch.tensor(sim), 1e-1), f"{result} != {sim}"


# Test fixtures
@pytest.fixture
def calculator():
    return ClassAwareExtraCalculations()


@pytest.fixture
def mock_model():
    model = Mock()
    model.prototype_layer = Mock()
    model.prototype_layer.prototype_class_identity = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    model.prototype_class_identity = model.prototype_layer.prototype_class_identity
    return model


@pytest.fixture
def mock_model_without_identity():
    model = Mock()
    model.prototype_layer = Mock()
    # Explicitly remove the attribute
    del model.prototype_layer.prototype_class_identity
    return model


# Test cases
def test_calculation_with_valid_model(calculator, mock_model):
    # Setup test data
    batch_size = 2
    num_prototypes = 3
    target = torch.tensor([0, 1])  # Two samples, classes 0 and 1

    # similarity_scores: (batch_size, num_prototypes)
    similarity_scores = torch.tensor(
        [
            [0.8, 0.3, 0.2],  # First sample's similarity to each prototype
            [0.4, 0.9, 0.1],  # Second sample's similarity to each prototype
        ]
    )

    # prototype_class_identity: (num_prototypes, num_classes) - one hot encoded
    mock_model.prototype_class_identity = torch.tensor(
        [
            [1, 0, 0],  # Prototype 0 belongs to class 0
            [0, 1, 0],  # Prototype 1 belongs to class 1
            [0, 0, 1],  # Prototype 2 belongs to class 2
        ]
    )

    # get_prototype_class_identity returns (batch_size, num_prototypes)
    mock_model.get_prototype_class_identity.return_value = torch.tensor(
        [
            [1, 0, 0],  # For first sample (class 0): only prototype 0 is correct
            [0, 1, 0],  # For second sample (class 1): only prototype 1 is correct
        ]
    ).t()  # Transposed to match the expected shape

    # Call the function
    result = calculator(
        target=target,
        model=mock_model,
        similarity_score_to_each_prototype=similarity_scores,
    )

    # Verify the results
    assert "prototype_class_identity" in result
    assert torch.equal(
        result["prototype_class_identity"], mock_model.prototype_class_identity
    )

    assert "prototypes_of_correct_class" in result
    expected_correct_class = torch.tensor(
        [
            [1, 0, 0],  # First sample (class 0): only prototype 0 is correct
            [0, 1, 0],  # Second sample (class 1): only prototype 1 is correct
        ]
    )
    assert torch.equal(result["prototypes_of_correct_class"], expected_correct_class)

    assert "prototypes_of_wrong_class" in result
    expected_wrong_class = 1 - expected_correct_class
    assert torch.equal(result["prototypes_of_wrong_class"], expected_wrong_class)

    assert "incorrect_class_prototype_activations" in result
    # For first sample: max of wrong prototypes ([0.3, 0.2]) = 0.3
    # For second sample: max of wrong prototypes ([0.4, 0.1]) = 0.4
    expected_incorrect_activations = torch.tensor([0.3, 0.4])
    assert torch.allclose(
        result["incorrect_class_prototype_activations"],
        expected_incorrect_activations,
        rtol=1e-5,
    )


def test_calculation_with_model_without_identity(
    calculator, mock_model_without_identity
):
    target = torch.tensor([0, 1])
    similarity_scores = torch.ones((2, 3))

    result = calculator(
        target=target,
        model=mock_model_without_identity,
        similarity_score_to_each_prototype=similarity_scores,
    )

    # Should return empty dictionary when prototype_class_identity is not present
    assert result == {}


def test_calculation_with_single_sample(calculator, mock_model):
    # Test with batch size of 1
    target = torch.tensor([0])  # Single sample of class 0

    # similarity_scores: (1, num_prototypes)
    similarity_scores = torch.tensor(
        [[0.8, 0.3, 0.2]]
    )  # One sample's similarity to each prototype

    # prototype_class_identity remains (num_prototypes, num_classes)
    mock_model.prototype_class_identity = torch.tensor(
        [
            [1, 0, 0],  # Prototype 0 belongs to class 0
            [0, 1, 0],  # Prototype 1 belongs to class 1
            [0, 0, 1],  # Prototype 2 belongs to class 2
        ]
    )

    # get_prototype_class_identity returns (num_prototypes, 1) after transpose
    mock_model.get_prototype_class_identity.return_value = torch.tensor([[1, 0, 0]]).t()

    result = calculator(
        target=target,
        model=mock_model,
        similarity_score_to_each_prototype=similarity_scores,
    )

    assert "incorrect_class_prototype_activations" in result
    # max of wrong prototypes ([0.3, 0.2]) = 0.3
    expected_incorrect_activations = torch.tensor([0.3])
    assert torch.allclose(
        result["incorrect_class_prototype_activations"],
        expected_incorrect_activations,
        rtol=1e-5,
    )


def test_calculation_edge_cases(calculator, mock_model):
    # Test with zero similarity scores
    target = torch.tensor([0])
    similarity_scores = torch.zeros((1, 3))

    mock_model.get_prototype_class_identity.return_value = torch.tensor([[1, 0, 0]])

    result = calculator(
        target=target,
        model=mock_model,
        similarity_score_to_each_prototype=similarity_scores,
    )

    assert "incorrect_class_prototype_activations" in result
    expected_incorrect_activations = torch.tensor([0.0])
    assert torch.allclose(
        result["incorrect_class_prototype_activations"],
        expected_incorrect_activations,
        rtol=1e-5,
    )
