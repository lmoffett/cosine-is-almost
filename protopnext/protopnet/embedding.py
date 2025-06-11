import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EmbeddedBackbone(nn.Module):
    """
    This is as backbone adapter for the original ProtoPNet implementation. It is used to wrap the backbone
    model and provide a common interface for the ProtoPNet architecture.
    """

    def __init__(self, embedded_model, input_channels: int = (3, 224, 224)):
        super(EmbeddedBackbone, self).__init__()
        self.embedded_model = embedded_model
        self.input_channels = input_channels

        with torch.no_grad():
            self.latent_dimension = self.__latent_dimension()

    def forward(self, x: torch.Tensor):
        # Define the forward pass for the backbone
        return self.embedded_model(x)

    def __latent_dimension(self):
        """
        The latent dimension for each input (without the batch dimension). For example, if the backbone is a ResNet-18,
            then the latent dimension would be (512, 7, 7).

        Returns: latent_dimension (tuple): The latent dimension for each input (without the batch dimension).
        """
        dummy_tensor = torch.randn(1, *self.input_channels)
        return self.embedded_model(dummy_tensor).shape[1:]

    def __repr__(self):
        return f"EmbeddedBackbone({self.embedded_model})"


class AddonLayers(nn.Module):
    """
    This is an implementation of the optional add-on layers for a ProtoPNet, which lies between the
    backbone and the prototype prediction head
    """

    def __init__(
        self,
        num_prototypes: torch.Tensor,
        input_channels: int = 512,
        proto_channels: int = None,
        proto_channel_multiplier: float = None,
        num_addon_layers: int = 2,
    ):
        super(AddonLayers, self).__init__()

        self.num_prototypes = num_prototypes

        self.input_channels = input_channels

        if proto_channel_multiplier is not None and proto_channels is not None:
            raise ValueError(
                "Cannot specify both proto_channel_multiplier and proto_channels"
            )
        elif proto_channel_multiplier is not None:
            self.proto_channels = int(proto_channel_multiplier * input_channels)

        elif proto_channels is not None:
            self.proto_channels = proto_channels
        else:
            self.proto_channels = input_channels

        if num_addon_layers == 0:
            if self.proto_channels != input_channels:
                logger.warning(
                    f"""Prototype channels are configured to be {self.proto_channels}, but there are 0 add-on layers and the input dimension is {input_channels}. Ignoring the prototype channel configuration; input_channels will be used for prototypes."""
                )
            self.add_on_layers = nn.Identity()
            self.proto_channels = input_channels

        else:
            self._build_layers(num_addon_layers)

    def _build_layers(self, num_addon_layers: int):
        mid_layers = []
        for _ in range(num_addon_layers - 1):
            mid_layers = mid_layers + [
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.proto_channels,
                    out_channels=self.proto_channels,
                    kernel_size=1,
                ),
            ]
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.proto_channels,
                kernel_size=1,
            ),
            *mid_layers,
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        # Define the forward pass for the backbone
        return self.add_on_layers(x)


class LightweightAddonLayers(AddonLayers):

    def __init__(
        self,
        num_prototypes,
        input_channels=512,
        proto_channels=None,
        proto_channel_multiplier=None,
        num_addon_layers=1,
        is_upsampled=False,
    ):

        if num_addon_layers != 1:
            logger.warning(
                f"""Lightweight add-on layer only supports 1 layer
                configuration, but {num_addon_layers} were given.
                This setting will be ignored."""
            )

        self.is_upsampled = is_upsampled

        super().__init__(
            num_prototypes,
            input_channels,
            proto_channels,
            proto_channel_multiplier,
            num_addon_layers,
        )

    def _build_layers(self, num_addon_layers: int):
        self.add_on_layers = nn.Sequential(
            (
                nn.Upsample(scale_factor=2, mode="bilinear")
                if self.is_upsampled
                else nn.Identity()
            ),
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=self.proto_channels,
                kernel_size=1,
            ),
            nn.Conv2d(
                in_channels=self.proto_channels,
                out_channels=self.proto_channels,
                kernel_size=1,
            ),
        )
