import pytest
import torch

from protopnet.pixel_space import (
    BilinearUpsampleActivations,
    PrototypePatchMask,
    SobelEdgeMask,
    upsample_mask,
    widen_mask,
)
from protopnet.prototype_layers import PrototypeFromSampleMeta


def test_upsample_mask_even_upsample():
    # Test a simple 2x2x2 mask upsampled by a factor of 2
    mask = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.int64)

    assert mask.shape == (2, 2, 2), "Mask should have shape (2, 2, 2)"

    upsampled = upsample_mask(mask, [4, 4])
    expected = torch.tensor(
        [
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(upsampled, expected), f"{upsampled} \n!=\n {expected}"


def test_upsample_mask_uneven_size():
    # Test a simple 2x2x2 mask upsampled by a factor of 2
    mask = torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]], dtype=torch.int64)

    assert mask.shape == (2, 2, 2), "Mask should have shape (2, 2, 2)"

    upsampled = upsample_mask(mask, [3, 5])
    expected = torch.tensor(
        [
            [[0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0]],
            [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(upsampled, expected), f"{upsampled} \n!=\n {expected}"


def test_widen_mask_single_step():
    # Test widening by 1 step
    mask = torch.tensor(
        [[[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]],
        dtype=torch.float32,
    )

    widened = widen_mask(mask, buffer_size=1)
    expected = torch.tensor(
        [[[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]],
        dtype=torch.uint8,
    )

    assert torch.equal(widened, expected), f"{widened} \n!=\n {expected}"


def test_widen_mask_multiple_steps():
    # Test widening by 2 steps
    mask = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.float32,
    )

    widened = widen_mask(mask, buffer_size=2)
    expected = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0],
            ]
        ],
        dtype=torch.uint8,
    )

    assert torch.equal(widened, expected), f"{widened} \n!=\n {expected}"


def test_widen_mask_beyond_borders():
    # Test widening stops at the border of the mask
    mask = torch.tensor(
        [
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
        ],
        dtype=torch.float32,
    )

    widened = widen_mask(mask, buffer_size=1)
    expected = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(widened, expected), f"{widened} \n!=\n {expected}"


class MockProtoPNet:
    """Mock class for ProtoPNet to use in testing PrototypePatchMask."""

    def __init__(self, prototype_layer):
        self.prototype_layer = prototype_layer
        self.prototype_layer.sample_id_to_prototype_index = {10: 0, 20: 1}
        self.prototype_layer.prototype_meta = [
            PrototypeFromSampleMeta(
                sample_id=10,
                target=1,
                hash="10",
                latent_patches=torch.tensor([[1, 0], [0, 0]]),
            ),
            PrototypeFromSampleMeta(
                sample_id=20,
                target=2,
                hash="20",
                latent_patches=torch.tensor([[0, 0], [0, 1]]),
            ),
        ]


@pytest.fixture
def prototype_patch_mask_with_model():
    """Fixture for the PrototypePatchMask with a mocked ProtoPNet model."""
    prototype_layer = type(
        "MockPrototypeLayer",
        (),
        {"sample_id_to_prototype_index": {1: 0, 2: 1}},  # Mock mapping
    )()
    model = MockProtoPNet(prototype_layer)
    return PrototypePatchMask(model, widen_prototype_pixels=1)


def test_invalid_widen_prototype_pixels():
    """Test that invalid widen_prototype_pixels raises a ValueError."""
    prototype_layer = type("MockPrototypeLayer", (), {})()
    model = MockProtoPNet(prototype_layer)

    with pytest.raises(
        ValueError, match="widen_prototype_pixels must be a non-negative integer"
    ):
        PrototypePatchMask(model, widen_prototype_pixels=-1)


def test_forward_invalid_sample_id(prototype_patch_mask_with_model):
    """Test that providing an invalid sample_id raises a KeyError."""
    img = torch.rand(1, 3, 8, 8)  # Mock image tensor
    invalid_sample_id = 999  # Invalid sample_id not in sample_id_to_prototype_index

    with pytest.raises(KeyError):
        prototype_patch_mask_with_model.forward(
            {"img": [img], "sample_id": [invalid_sample_id]}
        )


def test_forward_masking(prototype_patch_mask_with_model):
    """Test that the forward method returns a tensor with the same shape as the input image."""
    img = torch.randn(2, 3, 5, 5) + 10  # Mock image tensor
    sample_id = [10, 20]  # Valid sample ID

    output = prototype_patch_mask_with_model.forward(
        {"img": img, "sample_id": sample_id}
    )
    img_out = output["img"]

    assert img_out.shape == img.shape, "Output shape should match input image shape."

    # the upper-left 4x4 matrix should be unmasked, but the last column and row should be masked
    assert torch.all(
        img_out[0, :, 0:4, 0:4] > 5
    ), "Values should be unmasked based on initialization"
    assert torch.all(
        img_out[0, :, 4, :] < 3
    ), f"Values should be masked: {output[0, :, 4, :]}"
    assert torch.all(
        img_out[0, :, :, 4] < 3
    ), f"Values should be masked: {output[0, :, :, 4]}"

    # the lower-right 3x3 matrix should be unmasked, but the last column and row should be masked
    assert torch.all(
        img_out[1, :, 2:5, 2:5] > 5
    ), "Values should be unmasked based on initialization"
    assert torch.all(
        img_out[1, :, 0:1, :] < 3
    ), f"Values should be masked: {output[1, :, 0, :]}"
    assert torch.all(
        img_out[1, :, :, 0:1] < 3
    ), f"Values should be masked: {output[1, :, :, 0]}"


def test_forward_from_batch():
    """Test that the forward method returns a tensor with the same shape as the input image."""
    img = torch.randn(2, 3, 5, 5) + 10  # Mock image tensor

    prototype_patch_mask = PrototypePatchMask()
    patch_mask = torch.tensor([[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.int8)

    output = prototype_patch_mask.forward({"img": img, "patch_maps": patch_mask})
    img_out = output["img"]

    assert img_out.shape == img.shape, "Output shape should match input image shape."

    # the upper-left 4x4 matrix should be unmasked, but the last column and row should be masked
    assert torch.all(
        img_out[0, :, 0:3, 0:3] > 5
    ), "Values should be unmasked based on initialization"
    assert torch.all(
        img_out[0, :, 3:4, :] < 3
    ), f"Values should be masked: {output[0, :, 4, :]}"
    assert torch.all(
        img_out[0, :, :, 3:4] < 3
    ), f"Values should be masked: {output[0, :, :, 4]}"

    # the lower-right 3x3 matrix should be unmasked, but the last column and row should be masked
    assert torch.all(
        img_out[1, :, 3:5, 3:5] > 5
    ), "Values should be unmasked based on initialization"
    assert torch.all(
        img_out[1, :, 0:3, :] < 3
    ), f"Values should be masked: {output[1, :, 0, :]}"
    assert torch.all(
        img_out[1, :, :, 0:3] < 3
    ), f"Values should be masked: {output[1, :, :, 0]}"


@pytest.fixture
def sobel_edge_mask():
    """Fixture to initialize SobelEdgeMask."""
    return SobelEdgeMask()


def test_sobel_edge_mask_initialization(sobel_edge_mask):
    """Test if SobelEdgeMask initializes correctly."""
    assert isinstance(
        sobel_edge_mask.sobel_x, torch.Tensor
    ), "sobel_x should be a tensor"
    assert isinstance(
        sobel_edge_mask.sobel_y, torch.Tensor
    ), "sobel_y should be a tensor"
    assert sobel_edge_mask.sobel_x.shape == (1, 1, 3, 3), "sobel_x has incorrect shape"
    assert sobel_edge_mask.sobel_y.shape == (1, 1, 3, 3), "sobel_y has incorrect shape"


def test_sobel_edge_mask_call_single_image(sobel_edge_mask):
    """Test SobelEdgeMask on a single grayscale image."""
    input_tensor = torch.ones(
        1, 1, 5, 5
    )  # Single grayscale image (batch_size=1, channels=1)
    result = sobel_edge_mask(input_tensor)
    assert result.shape == input_tensor.shape, "Output shape does not match input shape"
    assert torch.all(
        (0 <= result) & (result <= 1)
    ), "Output values should be clipped between 0 and 1"


def test_sobel_edge_mask_call_batch_images(sobel_edge_mask):
    """Test SobelEdgeMask on a batch of grayscale images."""
    input_tensor = torch.rand(4, 1, 10, 10)  # Batch of 4 grayscale images
    result = sobel_edge_mask(input_tensor)
    assert result.shape == input_tensor.shape, "Output shape does not match input shape"
    assert torch.all(
        (0 <= result) & (result <= 1)
    ), "Output values should be clipped between 0 and 1"


def test_sobel_edge_mask_known_input(sobel_edge_mask):
    """Test SobelEdgeMask on a known input and output."""
    input_tensor = torch.tensor(
        [
            [
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            ]
        ],
        dtype=torch.float32,
    )  # Shape: (1, 1, 4, 4)

    result = sobel_edge_mask(input_tensor)

    expected_output = torch.tensor(
        [
            [
                [
                    [1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(
        result, expected_output, atol=1e-5
    ), "Output does not match expected result"


def test_sobel_edge_mask_invalid_input(sobel_edge_mask):
    """Test SobelEdgeMask with invalid input shapes."""
    with pytest.raises(AssertionError, match="Expected batch of shape "):
        # Missing batch dimension
        sobel_edge_mask(torch.rand(3, 10, 10))


def test_upsampling_to_target_size():
    """
    Test that the upsampling function correctly resizes the input to the target dimensions.
    """
    # Setup
    input_tensor = torch.randn(
        1, 3, 16, 16
    )  # batch_size=1, channels=3, height=16, width=16
    target_size = (32, 32)
    upsampler = BilinearUpsampleActivations(target_size)

    # Execute
    output = upsampler(input_tensor)

    assert (
        "upsampled_activation" in output
    ), "Output should contain 'upsampled_activations'"
    upsampled_activation = output["upsampled_activation"]

    # Assert
    assert upsampled_activation.shape == (
        1,
        3,
        32,
        32,
    ), f"Expected shape (1, 3, 32, 32), got {output.shape}"
    assert torch.is_floating_point(
        upsampled_activation
    ), "Output should maintain float dtype"


def test_batch_dimension_handling():
    """
    Test that the upsampling function correctly handles multiple images in a batch.
    """
    # Setup
    batch_size = 4
    input_tensor = torch.randn(
        batch_size, 2, 8, 8
    )  # batch_size=4, channels=2, height=8, width=8
    target_size = (12, 12)
    upsampler = BilinearUpsampleActivations(target_size)

    # Execute
    output = upsampler(input_tensor)

    assert (
        "upsampled_activation" in output
    ), "Output should contain 'upsampled_activations'"
    upsampled_activation = output["upsampled_activation"]

    # Assert
    assert upsampled_activation.shape == (
        batch_size,
        2,
        12,
        12,
    ), f"Expected shape ({batch_size}, 2, 12, 12), got {output.shape}"
    assert (
        upsampled_activation.shape[0] == batch_size
    ), "Batch dimension should be preserved"
