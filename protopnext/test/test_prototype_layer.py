from typing import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from protopnet.activations import ConvolutionalSharedOffsetPred, CosPrototypeActivation
from protopnet.prototype_layers import (
    ClassAwarePrototypeLayer,
    DeformablePrototypeLayer,
    PrototypeFromSampleMeta,
    PrototypeFromSampleSource,
    PrototypeLayer,
    PrototypeRandomInitMeta,
    PrototypeWithMeta,
    latent_prototype_patch_map,
)


# Example: Define a simple activation function for testing purposes
def mock_activation(x):
    return x


# Test PrototypeLayer
def test_prototype_layer_initialization():
    # Create an instance of PrototypeLayer
    layer = PrototypeLayer(
        num_prototypes=5,
        activation_function=mock_activation,
        latent_channels=128,
        prototype_dimension=(2, 2),
    )

    # Check attributes
    assert layer.num_prototypes == 5
    assert layer.activation_function == mock_activation
    assert layer.latent_channels == 128
    assert layer.prototype_dimension == (2, 2)
    assert isinstance(layer._prototype_meta, list)
    assert len(layer._prototype_meta) == 5
    assert layer._sample_id_to_prototype_indices == {}


def test_prototype_layer_save_load_state():
    # Initialize a layer and modify some attributes
    layer = PrototypeLayer(
        num_prototypes=3,
        activation_function=mock_activation,
        latent_channels=64,
        prototype_dimension=(1, 1),
    )
    # Modify internal states
    layer._sample_id_to_prototype_indices = {"sample1": {0}}
    layer._prototype_meta[0] = PrototypeFromSampleMeta(
        sample_id=b"test_id",
        target=1,
        hash=b"hash_value",
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    # Save state_dict
    state_dict = layer.state_dict()

    # Load into a new instance
    loaded_layer = PrototypeLayer(
        num_prototypes=3,
        activation_function=mock_activation,
        latent_channels=64,
        prototype_dimension=(1, 1),
    )
    loaded_layer.load_state_dict(state_dict)

    # Check if state is restored
    assert loaded_layer.num_prototypes == layer.num_prototypes
    assert loaded_layer.latent_channels == layer.latent_channels
    assert (
        loaded_layer._sample_id_to_prototype_indices
        == layer._sample_id_to_prototype_indices
    )
    assert loaded_layer._prototype_meta[0] == layer._prototype_meta[0]


def test_prototype_layer_save_load_state_from_file(temp_dir):
    # Initialize a layer and modify some attributes
    layer = PrototypeLayer(
        num_prototypes=15,
        activation_function=mock_activation,
        latent_channels=32,
        prototype_dimension=(2, 2),
    )
    # Modify internal states
    layer._sample_id_to_prototype_indices = {"sample1": {0}}
    layer._prototype_meta[0] = PrototypeFromSampleMeta(
        sample_id=b"test_id",
        target=1,
        hash=b"hash_value",
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    torch.save(layer, temp_dir / "test_layer.pth")
    loaded_layer = torch.load(temp_dir / "test_layer.pth")

    # Check if state is restored
    assert loaded_layer.num_prototypes == layer.num_prototypes
    assert loaded_layer.latent_channels == layer.latent_channels
    assert (
        loaded_layer._sample_id_to_prototype_indices
        == layer._sample_id_to_prototype_indices
    )
    assert loaded_layer._prototype_meta[0] == layer._prototype_meta[0]


@pytest.mark.parametrize(
    "prototype_layer", [ClassAwarePrototypeLayer, DeformablePrototypeLayer]
)
def test_class_aware_prototype_layer_save_load_state_from_file(
    temp_dir, prototype_layer
):
    # Initialize a layer and modify some attributes
    extra_args = {}
    if prototype_layer == DeformablePrototypeLayer:
        extra_args = {
            "epsilon_val": 1e-8,
            "offset_predictor": ConvolutionalSharedOffsetPred(
                prototype_shape=(3, 3, 1, 1)
            ),
        }
    layer = prototype_layer(
        prototype_class_identity=torch.eye(3),
        activation_function=mock_activation,
        latent_channels=32,
        prototype_dimension=(2, 2),
        **extra_args,
    )
    # Modify internal states
    layer._sample_id_to_prototype_index = {"sample1": {0}}
    layer._prototype_meta[0] = PrototypeFromSampleMeta(
        sample_id=b"test_id",
        target=1,
        hash=b"hash_value",
        latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    )

    torch.save(layer, temp_dir / "test_class_aware_layer.pth")
    loaded_layer = torch.load(temp_dir / "test_class_aware_layer.pth")

    # Check if state is restored
    assert torch.all(
        loaded_layer.prototype_class_identity == layer.prototype_class_identity
    )
    assert loaded_layer.num_classes == layer.num_classes
    assert loaded_layer.num_prototypes_per_class == layer.num_prototypes_per_class
    assert loaded_layer.class_specific_project == layer.class_specific_project
    assert loaded_layer.latent_channels == layer.latent_channels
    assert (
        loaded_layer._sample_id_to_prototype_indices
        == layer._sample_id_to_prototype_indices
    )
    assert loaded_layer._prototype_meta[0] == layer._prototype_meta[0]

    if prototype_layer == DeformablePrototypeLayer:
        assert loaded_layer.epsilon_val == layer.epsilon_val
        assert loaded_layer.offset_predictor == layer.offset_predictor


@pytest.mark.parametrize(
    "offset_init,contiguous_patches",
    [(torch.nn.init.kaiming_normal_, False), (torch.nn.init.zeros_, True)],
)
def test_deformable_masking(offset_init, contiguous_patches):
    """
    For this test, we'll want to:
    1) Project a deformable prototype layer with a single latent image
    2) Run a forward pass of this prototype layer on the
        latent image it was projected to
    3) Run a forward pass of the prototype layer on the latent image
        with our patch mask applied, and with offsets passed in explicitly
    4) Confirm that the output is identical for the two
    """

    protos_per_class = 10
    prototype_class_identity = torch.tensor([[1, 0]] * protos_per_class)

    cosine_activation = CosPrototypeActivation()

    prototype_layer = DeformablePrototypeLayer(
        activation_function=cosine_activation,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=(2, 2),
        latent_channels=32,
    )
    prototype_layer.offset_predictor.offset_predictor
    offset_init(prototype_layer.offset_predictor.offset_predictor.weight)

    latent_input = torch.randn(1, prototype_layer.prototype_tensors.shape[1], 10, 10)

    global_max_proto_act = torch.full(
        (prototype_class_identity.shape[0],), -float("inf")
    )
    global_max_fmap_patches = torch.zeros_like(prototype_layer.prototype_tensors)
    global_prototype_source = list(prototype_layer.prototype_meta)
    prototype_layer.update_prototypes_on_batch(
        latent_input,
        0,
        global_max_proto_act,
        global_max_fmap_patches,
        global_prototype_source,
        torch.tensor([0]),
        torch.tensor([0]),
    )

    num_contiguous = 0
    prototype_with_meta = []
    for proto_tensor, source in zip(global_max_fmap_patches, global_prototype_source):
        prototype_with_meta.append(
            PrototypeWithMeta(prototype_tensor=proto_tensor, meta=source.as_meta())
        )
        num_contiguous += int(source.contiguous_patches)

    if contiguous_patches:
        assert num_contiguous == len(prototype_with_meta)
    else:
        # at least one should not be contiguous
        assert num_contiguous < len(prototype_with_meta)

    prototype_layer.set_prototypes(prototype_with_meta)

    for p in range(protos_per_class):
        cur_proto_metadata = global_prototype_source[p]
        sample_locs = cur_proto_metadata.sample_locations.unsqueeze(0)
        masked_latent_input = (1 * cur_proto_metadata.latent_patches) * latent_input

        unmasked_output_post_proj = prototype_layer(latent_input, sample_locs)[
            "prototype_activations"
        ]

        masked_output_post_proj = prototype_layer(masked_latent_input, sample_locs)[
            "prototype_activations"
        ]
        assert torch.allclose(
            masked_output_post_proj[:, p].max(), unmasked_output_post_proj[:, p].max()
        ), (masked_output_post_proj[:, p].max(), unmasked_output_post_proj[:, p].max())


# Test PrototypeSourceSampleMeta
def test_prototype_source_sample_meta_equality():
    # Create two instances with the same data
    meta1 = PrototypeFromSampleMeta(
        sample_id=b"sample_id",
        target=1,
        hash=b"hash1",
        latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    )
    meta2 = PrototypeFromSampleMeta(
        sample_id=b"sample_id",
        target=1,
        hash=b"hash1",
        latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
    )

    # Check equality
    assert meta1 == meta2


def test_prototype_source_sample_meta_binary_constraint():
    # Create instance with binary latent_patches
    PrototypeFromSampleMeta(
        sample_id=b"test_id",
        target=1,
        hash=b"hash",
        latent_patches=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 1]]),
    )

    # Create instance with non-binary latent_pat


def test_prototype_density_metrics_class_agnostic():
    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = PrototypeLayer(
        activation_function=cosine_activation,
        num_prototypes=6,
        prototype_dimension=(2, 2),
    )

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 14, 14))

    prototype_layer.set_prototypes(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototypes(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototypes(proto_tensors)
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'


@pytest.mark.parametrize(
    "proto_layer", [DeformablePrototypeLayer, ClassAwarePrototypeLayer]
)
def test_prototype_density_metrics_class_aware(proto_layer):
    num_classes = 3
    prototype_class_identity = torch.randn((num_classes * 2, 3))

    # Test construction with cosine activation ----------------
    cosine_activation = CosPrototypeActivation()

    prototype_layer = proto_layer(
        activation_function=cosine_activation,
        prototype_class_identity=prototype_class_identity,
        prototype_dimension=(2, 2),
    )

    proto_tensors = torch.ones_like(prototype_layer.prototype_tensors)

    # Running forward to compute latent space size
    prototype_layer(torch.randn(1, proto_tensors.shape[1], 3, 3))

    def metadatafy_tensors(tensors):
        return [
            PrototypeWithMeta(
                prototype_tensor=tensors[i],
                meta=PrototypeFromSampleMeta(
                    sample_id=b"test_id",
                    target=i % num_classes,
                    hash=b"hash",
                    latent_patches=torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                ),
            )
            for i in range(tensors.shape[0])
        ]

    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 1
    ), f'Error: With only 1 unique part, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 1
    ), f'Error: With only 1 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[0, :, :, :] = torch.randn(*proto_tensors[0, :, :, :].shape)
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes, reported {unique_proto_stats["n_unique_protos"]} prototypes'

    proto_tensors[1, :, :, :] = proto_tensors[0, :, :, :] + 1e-12
    prototype_layer.set_prototypes(metadatafy_tensors(proto_tensors))
    unique_proto_stats = prototype_layer.get_prototype_complexity()

    assert (
        unique_proto_stats["n_unique_proto_parts"] == 5
    ), f'Error: With 5 unique parts and added floating point error, reported {unique_proto_stats["n_unique_proto_parts"]} parts'
    assert (
        unique_proto_stats["n_unique_protos"] == 2
    ), f'Error: With 2 unique prototypes and added floating point error, reported {unique_proto_stats["n_unique_protos"]} prototypes'


def test_prototype_source_sample_equality():
    # Case 1: Equal instances (identical metadata and tensors)
    sample1 = PrototypeFromSampleSource(
        sample_id=b"123",
        target=1,
        hash=b"abc",
        img=torch.tensor([1.0, 2.0, 3.0]),
        embedding=torch.tensor([4.0, 5.0, 6.0]),
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    sample2 = PrototypeFromSampleSource(
        sample_id=b"123",
        target=1,
        hash=b"abc",
        img=torch.tensor([1.0, 2.0, 3.0]),
        embedding=torch.tensor([4.0, 5.0, 6.0]),
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    assert (
        sample1 == sample2
    ), "Instances with identical metadata and tensors should be equal."

    # Case 2: Non-equal instances (different orig tensor)
    sample_diff_orig = PrototypeFromSampleSource(
        sample_id=b"123",
        target=1,
        hash=b"abc",
        img=torch.tensor([1.0, 2.0, 3.1]),
        embedding=torch.tensor([4.0, 5.0, 3.0]),
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    assert (
        sample1 != sample_diff_orig
    ), "Instances with different orig tensors should not be equal."

    # Case 3: Non-equal instances (different embedding tensor)
    sample_diff_embedding = PrototypeFromSampleSource(
        sample_id=b"123",
        target=1,
        hash=b"abc",
        img=torch.tensor([1.0, 2.0, 2.0]),
        embedding=torch.tensor([4.0, 3.0, 2.0]),
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    assert (
        sample1 != sample_diff_embedding
    ), "Instances with different embedding tensors should not be equal."

    # Case 4: Non-equal instances (different metadata)
    sample_diff_metadata = PrototypeFromSampleSource(
        sample_id=b"124",
        target=2,
        hash=b"xyz",
        img=torch.tensor([1.0, 2.0, 3.0]),
        embedding=torch.tensor([4.0, 5.0, 6.0]),
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    assert (
        sample1 != sample_diff_metadata
    ), "Instances with different metadata should not be equal."

    # Case 5: Comparison with different type
    other_type_instance = PrototypeFromSampleMeta(
        sample_id=b"123",
        target=1,
        hash=b"abc",
        latent_patches=torch.tensor([[1, 0], [0, 0]]),
    )

    assert (
        sample1 != other_type_instance
    ), "Instances of different types should not be equal."

    # Case 6: Different Proto Patches
    sample_diff_metadata = PrototypeFromSampleSource(
        sample_id=b"124",
        target=1,
        hash=b"xyz",
        img=torch.tensor([1.0, 2.0, 3.0]),
        embedding=torch.tensor([4.0, 5.0, 6.0]),
        latent_patches=torch.tensor([[0, 0], [0, 1]]),
    )


def test_latent_prototype_patch_map_2d_multi_proto_dim():
    latent_dim = (4, 5)
    prototype_locations = torch.tensor([[1, 3], [2, 4]])
    expected_output = torch.zeros(latent_dim)
    expected_output[1:3, 2:4] = 1

    output = latent_prototype_patch_map(latent_dim, prototype_locations)
    assert torch.equal(output, expected_output), "2D case with single location failed."


def test_latent_prototype_patch_map_2d_single_proto_dim():
    latent_dim = (4, 5)
    prototype_locations = torch.tensor([[3, 4], [1, 2]])
    expected_output = torch.zeros(latent_dim)
    expected_output[3:4, 1:2] = 1

    output = latent_prototype_patch_map(latent_dim, prototype_locations)
    assert torch.equal(
        output, expected_output
    ), "2D case with multiple locations failed."


def test_latent_prototype_patch_map_3d_single_location():
    latent_dim = (3, 4, 5)
    prototype_locations = torch.tensor([[1, 3], [2, 4], [0, 2]])
    expected_output = torch.zeros(latent_dim)
    expected_output[1:3, 2:4, 0:2] = 1

    output = latent_prototype_patch_map(latent_dim, prototype_locations)
    assert torch.equal(output, expected_output), "3D case with single location failed."


def test_prune_prototypes_by_index():
    # Create an instance of PrototypeLayer
    num_prototypes = 5
    layer = PrototypeLayer(
        num_prototypes=num_prototypes,
        activation_function=mock_activation,
        latent_channels=128,
        prototype_dimension=(1, 1),
    )

    remove_prototypes = [0, 1]

    # preform prune
    layer.prune_prototypes_by_index(remove_prototypes)

    # case 1: check if new count of prototypes
    assert layer.num_prototypes == (num_prototypes - len(remove_prototypes))

    # case 2: check if number of entries (regardless of `None` entries) in the
    # `prototype_meta` list are preserved
    assert len(layer.prototype_meta) == (num_prototypes - len(remove_prototypes))

    # case 3: check if `layer.sample_id_to_prototype_index` is updated correctly
    assert all(
        [idx not in layer.sample_id_to_prototype_indices for idx in remove_prototypes]
    )


@pytest.fixture
def class_agnostic_prototype_layer():
    layer = PrototypeLayer(
        num_prototypes=5,
        activation_function=mock_activation,
        latent_channels=4,
        prototype_dimension=(1, 1),
    )

    return layer


def test_set_prototypes_with_single_tensor(class_agnostic_prototype_layer):
    """Test setting a single prototype tensor"""
    layer = class_agnostic_prototype_layer
    proto_tensor = torch.randn(3, 4)  # Example dimensions
    layer.set_prototypes([proto_tensor])

    assert torch.equal(layer.prototype_tensors, proto_tensor.unsqueeze(0))
    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert len(layer._sample_id_to_prototype_indices) == 0


def test_set_prototypes_with_multiple_tensors(class_agnostic_prototype_layer):
    """Test setting multiple prototype tensors"""
    layer = class_agnostic_prototype_layer
    proto_tensors = [torch.randn(3, 4) for _ in range(3)]
    layer.set_prototypes(proto_tensors)

    assert torch.equal(layer.prototype_tensors, torch.stack(proto_tensors))
    assert len(layer._prototype_meta) == 3
    assert all(
        isinstance(meta, PrototypeRandomInitMeta) for meta in layer._prototype_meta
    )
    assert len(layer._sample_id_to_prototype_indices) == 0


def test_set_prototypes_with_prototype_with_meta(class_agnostic_prototype_layer):
    """Test setting a prototype with metadata"""
    layer = class_agnostic_prototype_layer

    # Create a PrototypeFromSampleMeta
    latent_patches = torch.ones(2, 2)  # Simple 2x2 binary tensor
    meta = PrototypeFromSampleMeta(
        sample_id=1, target=0, hash=b"test", latent_patches=latent_patches
    )

    proto_tensor = torch.randn(3, 4)
    proto_with_meta = PrototypeWithMeta(prototype_tensor=proto_tensor, meta=meta)

    layer.set_prototypes([proto_with_meta])

    assert torch.equal(layer.prototype_tensors, proto_tensor.unsqueeze(0))
    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeFromSampleMeta)
    assert layer._sample_id_to_prototype_indices[1] == {0}


def test_set_prototypes_mixed_types(class_agnostic_prototype_layer):
    """Test setting a mixture of tensors and prototypes with metadata"""
    layer = class_agnostic_prototype_layer

    # Create some regular tensors
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(3, 4)

    # Create a prototype with metadata
    latent_patches = torch.ones(2, 2)
    meta = PrototypeFromSampleMeta(
        sample_id=1, target=0, hash=b"test", latent_patches=latent_patches
    )
    proto_with_meta = PrototypeWithMeta(prototype_tensor=torch.randn(3, 4), meta=meta)

    # Mix them together
    prototypes = [tensor1, proto_with_meta, tensor2]
    layer.set_prototypes(prototypes)

    assert layer.prototype_tensors.shape == (3, 3, 4)
    assert len(layer._prototype_meta) == 3
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert isinstance(layer._prototype_meta[1], PrototypeFromSampleMeta)
    assert isinstance(layer._prototype_meta[2], PrototypeRandomInitMeta)
    assert layer._sample_id_to_prototype_indices[1] == {1}


def test_set_prototypes_with_random_init_meta(class_agnostic_prototype_layer):
    """Test setting prototypes with PrototypeRandomInitMeta"""
    layer = class_agnostic_prototype_layer

    # Create a prototype with random init metadata
    random_meta = PrototypeRandomInitMeta(initialization_strategy="uniform")
    proto_with_meta = PrototypeWithMeta(
        prototype_tensor=torch.randn(3, 4), meta=random_meta
    )

    layer.set_prototypes([proto_with_meta])

    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert layer._prototype_meta[0].initialization_strategy == "uniform"
    assert len(layer._sample_id_to_prototype_indices) == 0


def test_set_prototypes_with_single_tensor(class_agnostic_prototype_layer):
    """Test setting a single prototype tensor"""
    layer = class_agnostic_prototype_layer
    proto_tensor = torch.randn(3, 4)  # Example dimensions
    layer.set_prototypes([proto_tensor])

    assert torch.equal(layer.prototype_tensors, proto_tensor.unsqueeze(0))
    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert len(layer._sample_id_to_prototype_indices) == 0


def test_set_prototypes_with_multiple_tensors(class_agnostic_prototype_layer):
    """Test setting multiple prototype tensors"""
    layer = class_agnostic_prototype_layer
    proto_tensors = [torch.randn(3, 4) for _ in range(3)]
    layer.set_prototypes(proto_tensors)

    assert torch.equal(layer.prototype_tensors, torch.stack(proto_tensors))
    assert len(layer._prototype_meta) == 3
    assert all(
        isinstance(meta, PrototypeRandomInitMeta) for meta in layer._prototype_meta
    )
    assert len(layer._sample_id_to_prototype_indices) == 0


def test_set_prototypes_with_prototype_with_meta(class_agnostic_prototype_layer):
    """Test setting a prototype with metadata"""
    layer = class_agnostic_prototype_layer

    # Create a PrototypeFromSampleMeta
    latent_patches = torch.ones(2, 2)  # Simple 2x2 binary tensor
    meta = PrototypeFromSampleMeta(
        sample_id=1, target=0, hash=b"test", latent_patches=latent_patches
    )

    proto_tensor = torch.randn(3, 4)
    proto_with_meta = PrototypeWithMeta(prototype_tensor=proto_tensor, meta=meta)

    layer.set_prototypes([proto_with_meta])

    assert torch.equal(layer.prototype_tensors, proto_tensor.unsqueeze(0))
    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeFromSampleMeta)
    assert layer._sample_id_to_prototype_indices[1] == {0}


def test_set_prototypes_mixed_types(class_agnostic_prototype_layer):
    """Test setting a mixture of tensors and prototypes with metadata"""
    layer = class_agnostic_prototype_layer

    # Create some regular tensors
    tensor1 = torch.randn(3, 4)
    tensor2 = torch.randn(3, 4)

    # Create a prototype with metadata
    latent_patches = torch.ones(2, 2)
    meta = PrototypeFromSampleMeta(
        sample_id=1, target=0, hash=b"test", latent_patches=latent_patches
    )
    proto_with_meta = PrototypeWithMeta(prototype_tensor=torch.randn(3, 4), meta=meta)

    # Mix them together
    prototypes = [tensor1, proto_with_meta, tensor2]
    layer.set_prototypes(prototypes)

    assert layer.prototype_tensors.shape == (3, 3, 4)
    assert len(layer._prototype_meta) == 3
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert isinstance(layer._prototype_meta[1], PrototypeFromSampleMeta)
    assert isinstance(layer._prototype_meta[2], PrototypeRandomInitMeta)
    assert layer._sample_id_to_prototype_indices[1] == {1}


def test_set_prototypes_with_random_init_meta(class_agnostic_prototype_layer):
    """Test setting prototypes with PrototypeRandomInitMeta"""
    layer = class_agnostic_prototype_layer

    # Create a prototype with random init metadata
    random_meta = PrototypeRandomInitMeta(initialization_strategy="uniform")
    proto_with_meta = PrototypeWithMeta(
        prototype_tensor=torch.randn(3, 4), meta=random_meta
    )

    layer.set_prototypes([proto_with_meta])

    assert len(layer._prototype_meta) == 1
    assert isinstance(layer._prototype_meta[0], PrototypeRandomInitMeta)
    assert layer._prototype_meta[0].initialization_strategy == "uniform"
    assert len(layer._sample_id_to_prototype_indices) == 0


@pytest.fixture
def class_aware_layer():
    # Create a simple prototype class identity tensor for 3 prototypes and 2 classes
    prototype_class_identity = torch.tensor([[1, 0], [1, 0], [0, 1]])

    return ClassAwarePrototypeLayer(
        activation_function=mock_activation,
        prototype_class_identity=prototype_class_identity,
        latent_channels=512,
        prototype_dimension=(1, 1),
    )


def test_class_aware_set_prototypes_basic(class_aware_layer):
    """Test basic functionality of setting prototypes in ClassAwarePrototypeLayer"""
    layer = class_aware_layer

    # Create prototypes with metadata
    latent_patches = torch.ones(2, 2)
    prototypes = []
    for i in range(3):
        meta = PrototypeFromSampleMeta(
            sample_id=i,
            target=0 if i < 2 else 1,  # First two prototypes class 0, last one class 1
            hash=f"test{i}".encode(),
            latent_patches=latent_patches,
        )
        proto_with_meta = PrototypeWithMeta(
            prototype_tensor=torch.randn(512, 1, 1), meta=meta
        )
        prototypes.append(proto_with_meta)

    layer.set_prototypes(prototypes)

    assert len(layer._prototype_meta) == 3
    assert all(
        isinstance(meta, PrototypeFromSampleMeta) for meta in layer._prototype_meta
    )
    assert layer._prototype_meta[0].target == 0
    assert layer._prototype_meta[1].target == 0
    assert layer._prototype_meta[2].target == 1

    assert layer.prototype_class_identity.shape == (3, 2)
    assert torch.equal(
        layer.prototype_class_identity, torch.tensor([[1, 0], [1, 0], [0, 1]])
    )


def test_class_aware_set_prototypes_invalid_type(class_aware_layer):
    """Test that setting raw tensors raises an error"""
    layer = class_aware_layer

    with pytest.raises(AttributeError):
        layer.set_prototypes([torch.randn(512, 1, 1)])
