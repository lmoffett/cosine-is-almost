from typing import Callable, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import Grayscale

from .prototypical_part_model import ProtoPNet


def upsample_mask(mask, target_shape: torch.Tensor):
    """
    Upsamples a batch of binary masks without interpolation, preserving rectangular shape.

    Args:
        mask (torch.Tensor): A binary mask tensor of shape (batch, *dims), where *dims
                            represents arbitrary dimensions.
        target_shape (*dims): The target shape to upsample the mask to.

    Returns:
        torch.Tensor: The upsampled mask, with the same shape as mask
    """

    # Ensure the mask is binary
    assert mask.unique().tolist() == [0, 1], "Mask should contain only 0s and 1s"

    # Expand the mask dimensions for nearest neighbor upsampling
    mask = mask.to(dtype=torch.float32)  # Convert to float to work with F.interpolate
    upsampled_mask = F.interpolate(mask.unsqueeze(1), size=target_shape, mode="nearest")

    # Squeeze out the extra dimension and convert back to binary
    upsampled_mask = upsampled_mask.squeeze(1).to(dtype=torch.int64)

    return upsampled_mask


def widen_mask(mask, buffer_size: int):
    """
    Widens a binary mask of ones by a specified number of steps in all directions.

    Args:
        mask (torch.Tensor): A binary mask tensor of shape (batch, height, width) or
                             (batch, depth, height, width) containing 0s and 1s.
        buffer_size (int): The number of steps to widen the mask in one direction.

    Returns:
        torch.Tensor: The widened mask, with ones expanded outward by the specified steps.
    """
    # Get the shape of the mask
    dims = mask.shape
    if len(dims) not in [3]:
        raise ValueError(
            "Expected mask of shape (batch, height, width) or (batch, depth, height, width)"
        )

    # Create an empty tensor for the widened mask
    widened_mask = torch.zeros_like(mask, dtype=mask.dtype)

    for b, this_mask in enumerate(mask):
        h_indices, w_indices = torch.nonzero(this_mask, as_tuple=True)
        h_min, h_max = h_indices.min().item(), h_indices.max().item()
        w_min, w_max = w_indices.min().item(), w_indices.max().item()

        # Expand the bounding box by buffer_size
        h_start = max(h_min - buffer_size, 0)
        h_end = min(h_max + buffer_size + 1, dims[1])
        w_start = max(w_min - buffer_size, 0)
        w_end = min(w_max + buffer_size + 1, dims[2])

        # Set the widened region to 1
        widened_mask[b, h_start:h_end, w_start:w_end] = 1

    return widened_mask


class GaussianNoiseMask:
    def __init__(self, mean: float = 0.5, std: float = 0.2):
        self.mean = mean
        self.std = std

    def __call__(self, batch: torch.Tensor):
        return (
            torch.randn(batch.shape, device=batch.device, requires_grad=False)
            * self.std
            + self.mean
        ).clip(0, 1)


class SobelEdgeMask:
    def __init__(self):
        self.sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        self.sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def __call__(self, batch: torch.Tensor):
        assert len(batch.shape) == 4, "Expected batch of shape (B, C, H, W)"

        batch = Grayscale()(batch)
        # Apply Sobel filters
        edges_x = F.conv2d(batch, self.sobel_x.to(batch.device), padding=1)
        edges_y = F.conv2d(batch, self.sobel_y.to(batch.device), padding=1)

        # Combine gradients
        clipped_grad = torch.sqrt(edges_x**2 + edges_y**2).clip(0, 1)
        return 1 - clipped_grad


class PrototypePatchMask(torch.nn.Module):
    def __init__(
        self,
        model: ProtoPNet = None,
        mask_generator: Callable = GaussianNoiseMask(),
        widen_prototype_pixels: int = 0,
    ):
        super(PrototypePatchMask, self).__init__()
        self.model: ProtoPNet = model
        self.mask_generator = mask_generator

        if widen_prototype_pixels < 0 or not isinstance(widen_prototype_pixels, int):
            raise ValueError("widen_prototype_pixels must be a non-negative integer.")
        self.widen_prototype_pixels = widen_prototype_pixels

    def forward(self, batch):
        """
        Given a B x C x H x W tensor, mask out all but the prototype patch for each sample associated with a prototype.
        If the sample_id is not in the model prototype metadata, an error is thrown.
        """
        output_dict = {**batch}
        img = batch["img"]

        if "patch_maps" in batch:
            patch_maps = batch["patch_maps"]
        else:
            sample_id = batch["sample_id"]
            # TODO: This should be generalized to be any calculation that runs on the batch
            sample_to_proto = self.model.prototype_layer.sample_id_to_prototype_index

            patch_maps = torch.stack(
                [
                    self.model.prototype_layer.prototype_meta[
                        sample_to_proto[s]
                    ].latent_patches.to(img.device)
                    for s in sample_id
                ]
            )

        patch_maps_upsampled = upsample_mask(patch_maps, img.shape[-2:])

        if self.widen_prototype_pixels > 0:
            patch_maps_upsampled = widen_mask(
                patch_maps_upsampled, self.widen_prototype_pixels
            )

        patch_maps_upsampled = patch_maps_upsampled.unsqueeze(1).expand_as(img)

        masked_img = self.mask_generator(img)

        patch_maps_inverse = patch_maps_upsampled == 0

        img_masked = masked_img * patch_maps_inverse + img * patch_maps_upsampled

        output_dict["img"] = img_masked

        return output_dict


class BilinearUpsampleActivations:
    """
    This implementation aligns with the configuration used in the original ProtoPNet.

    This is primarily used to change the size of activation maps to match them to the input pixel size for the purpose of pixel-wise loss terms or visualizations.
    """

    def __init__(self, image_size: Union[torch.Size, Tuple[int, int]]):
        """
        image_size (Union[torch.Size, Tuple[int, int]]): Target image size as (height, width).
        """
        self.image_size = image_size

    def __call__(self, prototype_activations: torch.Tensor):
        """
        Applies bilinear upsampling to the input activations.
        """
        return {
            "upsampled_activation": torch.nn.Upsample(
                size=(self.image_size[0], self.image_size[1]),
                mode="bilinear",
                align_corners=False,
            )(prototype_activations)
        }
