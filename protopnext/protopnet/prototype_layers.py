import collections
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import ConvolutionalSharedOffsetPred, CosPrototypeActivation
from .utilities.project_utilities import custom_unravel_index, hash_func


class PrototypeRandomInitMeta:
    """
    Marker class for prototypes that are randomly initialized.
    """

    def __init__(self, initialization_strategy="random"):
        self.initialization_strategy = initialization_strategy

    @property
    def source(self):
        return self.initialization_strategy


@dataclass
class PrototypeFromSampleMeta:
    """
    Contains the metadata needed to find the source of a prototype.
    """

    sample_id: Union[str, int]
    target: Union[int, torch.Tensor]
    hash: bytes
    """
    A binary tensor of the embedding dimensions indicating which patches of the latent space were used to generate the prototype.
    1 if the patch is part of the prototype, 0 if not.
    """
    latent_patches: torch.Tensor
    contiguous_patches: bool = field(init=False)

    @property
    def source(self):
        return "sample"

    def __post_init__(self):
        # ensure latent patches are binary, rectangular, and contiguous
        assert torch.all(
            (self.latent_patches == 0) | (self.latent_patches == 1)
        ), "Latent patches must be binary."

        indices = self.latent_patches.nonzero(as_tuple=True)

        min_coords = [dim_indices.min().item() for dim_indices in indices]
        max_coords = [dim_indices.max().item() for dim_indices in indices]

        # Create slices for each dimension
        slices = tuple(
            slice(min_coord, max_coord + 1)
            for min_coord, max_coord in zip(min_coords, max_coords)
        )

        # Slice the binary mask to extract the bounding box
        bound_box = self.latent_patches[slices]

        # Check if the entire bounding box is filled with True values
        if torch.all(bound_box) and bound_box.sum().item() == len(
            self.latent_patches.nonzero()
        ):
            self.contiguous_patches = True
        else:
            self.contiguous_patches = False

        self.bounding_box = list(zip(min_coords, max_coords))

        if isinstance(self.sample_id, torch.Tensor):
            assert self.sample_id.dtype in [
                torch.int,
                torch.long,
            ], "Sample ID must be an integer if it is provided as a tensor."
            self.sample_id = self.sample_id.item()

        if isinstance(self.target, torch.Tensor):
            assert self.target.dtype in [
                torch.int,
                torch.long,
            ], "Target must be an integer if it is provided as a tensor."
            self.target = self.target.item()

    def __eq__(self, other):
        # Check if the other object is of the same type
        if not isinstance(other, PrototypeFromSampleMeta):
            return False

        id_equal = self.sample_id == other.sample_id
        label_equal = self.target == other.target
        hash_equal = self.hash == other.hash
        latent_patches_equal = torch.equal(self.latent_patches, other.latent_patches)

        # Return True only if all comparisons are True
        return id_equal and label_equal and hash_equal and latent_patches_equal


@dataclass
class PrototypeFromSampleSource(PrototypeFromSampleMeta):
    """
    Contains the original image and embedding of a prototype alongside the metadata.
    """

    img: torch.Tensor
    embedding: torch.Tensor
    sample_locations: Optional[torch.Tensor] = None

    def as_meta(self):
        return PrototypeFromSampleMeta(
            sample_id=self.sample_id,
            target=self.target,
            hash=self.hash,
            latent_patches=self.latent_patches,
        )

    def __eq__(self, other):
        # Check if the other object is of the same type
        if not super().__eq__(other):
            return False

        if not isinstance(other, PrototypeFromSampleSource):
            return False

        orig_close = torch.allclose(self.img, other.img)
        embedding_close = torch.allclose(self.embedding, other.embedding)

        # Return True only if all comparisons are True
        return orig_close and embedding_close


@dataclass
class PrototypeWithMeta:
    """
    Wrapper for the prototype tensor and metadata.
    """

    prototype_tensor: torch.Tensor
    meta: Union[PrototypeRandomInitMeta, PrototypeFromSampleMeta]


@dataclass
class PrototypeWithSource:
    """
    Wrapper for the prototype tensor and all source data.
    """

    prototype_tensor: torch.Tensor
    source: PrototypeFromSampleSource


def latent_prototype_patch_map(latent_dim, prototype_locations, device=None):
    """
    Given a latent space dimension and prototype locations, creates a one-hot encoded tensor indicating the patches that generated the prototype.

    Args:
        latent_dim: The dimension of the latent space
        prototype_locations: latent_dim x 2 tensor of start/ends of the prototype patches
    """
    patch_map = torch.zeros(
        latent_dim, device=device, dtype=torch.bool, requires_grad=False
    )

    proto_slices = [slice(start, end) for start, end in prototype_locations]
    patch_map[proto_slices] = True

    return patch_map


def latent_prototype_patch_map_deformed(latent_dim, sample_locs, device=None):
    """
    Given a latent space dimension and prototype locations, creates a one-hot encoded tensor indicating the patches that generated the prototype.

    Args:
        latent_dim: The dimension of the latent space
        sample_locs: a (proto_h, proto_w, 2) tensor describing the
            location to compare each prototypical part to at the
            center location and image index of maximum similarity.
            Values are normalized in (-1, 1), with -1 indicating top/left
    """
    patch_map = torch.zeros(
        latent_dim, device=device, dtype=torch.bool, requires_grad=False
    )

    # Renormalize sample_locs to fall in (0, 1)
    sample_locs = (sample_locs + 1) / 2

    # While sometimes we can sample out of bounds, assert that
    # this usually doesn't happen
    # assert (1.0 * (sample_locs >= 0)).mean() > 0.9, sample_locs
    # assert (1.0 * (sample_locs <= 1)).mean() > 0.9, sample_locs

    for proto_h in range(sample_locs.shape[0]):
        for proto_w in range(sample_locs.shape[1]):
            y_for_part = sample_locs[proto_h, proto_w, 0] * (latent_dim[-1] - 1)
            y_for_part = min(y_for_part, torch.tensor(patch_map.shape[1] - 1))
            y_for_part = max(y_for_part, torch.tensor(0))
            x_for_part = sample_locs[proto_h, proto_w, 1] * (latent_dim[-2] - 1)
            x_for_part = min(x_for_part, torch.tensor(patch_map.shape[0] - 1))
            x_for_part = max(x_for_part, torch.tensor(0))

            patch_map[int(y_for_part.floor()), int(x_for_part.floor())] = True
            patch_map[int(y_for_part.floor()), int(x_for_part.ceil())] = True
            patch_map[int(y_for_part.ceil()), int(x_for_part.floor())] = True
            patch_map[int(y_for_part.ceil()), int(x_for_part.ceil())] = True

    return patch_map


class PrototypeLayer(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        activation_function: Callable,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        k_for_topk: int = 1,
        with_fa: bool = False,
    ):
        """
        Args:
            activation_function: The activation function to use
            latent_channels: The number of output channels from the backbone
            prototype_dimension: A (proto_h, proto_w) tuple indicating the size of
                the prototypes in the latent space
            k_for_topk: The number of prototypes to consider for each image
        """
        super(PrototypeLayer, self).__init__()
        self.activation_function = activation_function
        self.latent_channels = latent_channels
        self.prototype_dimension = prototype_dimension
        self.__initialize_random_prototypes(num_prototypes)

        # FIXME: This should not be a property of the Prototype layer
        self.with_fa = with_fa
        self.k_for_topk = k_for_topk

        # FIXME should this be deleted?
        self.latent_spatial_size = None

    def set_prototypes(
        self,
        prototypes: Union[torch.Tensor, List[Union[PrototypeWithMeta, torch.Tensor]]],
    ):
        """
        Sets the prototypes to the given list of prototypes.
        If a prototype is given as a tensor, the metadata is set to a PrototypeRandomInitMeta.
        """
        new_proto_tensors = []
        new_proto_meta = []
        sample_id_to_prototype_indices = collections.defaultdict(set)
        for i, proto in enumerate(prototypes):
            if isinstance(proto, torch.Tensor):
                new_proto_tensors.append(proto)
                new_proto_meta.append(PrototypeRandomInitMeta())
            else:
                new_proto_tensors.append(proto.prototype_tensor)
                new_proto_meta.append(proto.meta)

                if proto.meta.source == "sample":
                    sample_id_to_prototype_indices[proto.meta.sample_id].add(i)

        self.prototype_tensors.data = torch.tensor(
            torch.stack(new_proto_tensors), dtype=torch.float32
        ).to(self.prototype_tensors.device)
        self._prototype_meta = new_proto_meta
        self._sample_id_to_prototype_indices = sample_id_to_prototype_indices

    def __initialize_random_prototypes(self, num_prototypes):
        self.prototype_tensors = nn.Parameter(
            torch.rand(
                num_prototypes,
                self.latent_channels,
                *self.prototype_dimension,
                requires_grad=True,
            )
        )

        self._prototype_meta = [
            PrototypeRandomInitMeta() for _ in range(self.num_prototypes)
        ]
        self._sample_id_to_prototype_indices = collections.defaultdict(set)

    @property
    def num_prototypes(self) -> int:
        return self.prototype_tensors.shape[0]

    @property
    def prototype_meta(
        self,
    ) -> List[Union[PrototypeFromSampleMeta, PrototypeRandomInitMeta]]:
        """
        Get the prototype tensors and metadata for each prototype.
        """
        return list(self._prototype_meta)

    @property
    def sample_id_to_prototype_indices(self) -> Dict[Any, Set[int]]:
        """
        Given a sample_id, find all prototypes that were generated from that sample.
        If no prototypes were generated from a sample, the set will be empty.
        """
        return collections.defaultdict(set, self._sample_id_to_prototype_indices)

    @property
    def min_sparsity(self):
        return 1 + 1 / (self.latent_spatial_size[0] * self.latent_spatial_size[1])

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        # Reorganize so that we have a collection of prototype part vectors
        part_vectors = self.prototype_tensors.permute(0, 2, 3, 1).reshape(
            -1, self.prototype_tensors.shape[1]
        )
        n_unique_proto_parts = (
            torch.round(part_vectors, decimals=decimal_precision).unique(dim=0).shape[0]
        )

        # Repeat to get the number of unique prototype tensors
        stacked_proto_vectors = self.prototype_tensors.reshape(
            self.prototype_tensors.shape[0], -1
        )
        n_unique_protos = (
            torch.round(stacked_proto_vectors, decimals=decimal_precision)
            .unique(dim=0)
            .shape[0]
        )

        min_sparsity = self.min_sparsity

        prototype_sparsity = n_unique_protos + n_unique_proto_parts / (
            self.latent_spatial_size[0] * self.latent_spatial_size[1]
        )

        prototype_sparsity = min_sparsity / prototype_sparsity

        return {
            "n_unique_proto_parts": n_unique_proto_parts,
            "n_unique_protos": n_unique_protos,
            "prototype_sparsity": prototype_sparsity,
        }

    def forward(self, x: torch.Tensor):
        """
        Provides a prototype similarity for each image at each location. This results in a tensor of shape
        (batch_size, num_prototypes, latent_height, latent_width)
        """
        prototype_activations = self.activation_function(x, self.prototype_tensors)

        if not hasattr(self, "latent_spatial_size") or self.latent_spatial_size is None:
            self.latent_spatial_size = (
                prototype_activations.shape[-2],
                prototype_activations.shape[-1],
            )

        output_dict = {
            "prototype_activations": prototype_activations,
        }

        return output_dict

    def get_extra_state(self):
        return {
            "_prototype_meta": self._prototype_meta,
            "_sample_id_to_prototype_indices": self._sample_id_to_prototype_indices,
            "activation_function": self.activation_function,
            "latent_channels": self.latent_channels,
            "prototype_dimension": self.prototype_dimension,
        }

    def set_extra_state(self, state):
        self._prototype_meta = state["_prototype_meta"]
        self._sample_id_to_prototype_indices = state["_sample_id_to_prototype_indices"]
        self.activation_function = state["activation_function"]
        self.latent_channels = state["latent_channels"]
        self.prototype_dimension = state["prototype_dimension"]

    def random_sample_ids_for_prototypes(self, dataset) -> List[Union[str, int]]:
        """
        Given a dataset, returns a list of samples that could be used as prototypes.
        The indexes of the samples match any restrictions about samples that can in that slot.
        For this implementation, selection is done at ruan
        """
        # init samples is a list of samples to init
        init_samples = []
        labels = []  # just for checking
        sample_ids = []

        # continue until a sample for every position for every class has been found
        while len(init_samples) < self.num_prototypes:
            # Select random input
            random_int = random.randint(0, len(dataset) - 1)
            img = dataset[random_int]["img"]
            label = dataset[random_int]["target"]

            # place the img in the correct position
            init_samples.append(img)
            labels.append(label)
            sample_ids.append(dataset[random_int]["sample_id"])

        return init_samples, labels, sample_ids

    def rsample_init(self, outputs, labels, sample_ids=None):
        """
        Args:
            outputs <- (num_prototypes, C, H, W) sized output to use for prototype initialization
        """

        # proto height and width
        proto_H, proto_W = self.prototype_dimension

        assert outputs.shape[0] == self.num_prototypes

        protos_with_meta = [None] * self.num_prototypes
        for i in range(self.num_prototypes):
            H_start = random.randint(0, outputs.shape[2] - proto_H)
            W_start = random.randint(0, outputs.shape[3] - proto_W)

            # TODO: 3Dify
            H_end = H_start + proto_H
            W_end = W_start + proto_W

            protos_with_meta[i] = PrototypeWithMeta(
                prototype_tensor=outputs[i, :, H_start:H_end, W_start:W_end],
                meta=PrototypeFromSampleMeta(
                    sample_id=sample_ids[i],
                    target=labels[i],
                    hash=hash_func(outputs[i]),
                    # TODO: 3Dify
                    latent_patches=latent_prototype_patch_map(
                        outputs.shape[-2:],
                        [(H_start, H_end), (W_start, W_end)],
                        device=outputs.device,
                    ),
                ),
            )

        self.set_prototypes(protos_with_meta)

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        global_prototype_meta: List[PrototypeFromSampleSource],
        sample_ids,
        labels,
    ):
        """
        Args:
            protoL_input_torch: Embeddings (batch_size x feature_dim x latent_h x latent_w)
            start_index_of_search_batch: The index of the first image in the search batch
            global_max_proto_act: The maximum prototype activation value for each prototype (num_prototypes)
            global_max_fmap_patches: The maximum feature map patch for each prototype, global_max_proto_act (num_prototypes x feature_dim x latent_h x latent_w)
            sample_ids: The sample ids for each image in the batch
            labels: The labels for each image in the batch
        """
        prototype_layer_stride = 1
        updated_prototype_indices = {}

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device)
        )["prototype_activations"]

        prototype_shape = self.prototype_tensors.shape

        for j in range(self.num_prototypes):
            proto_act_j = proto_act_torch[:, j, :, :]
            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                fmap_height_end_index = fmap_height_start_index + prototype_shape[-2]
                fmap_width_end_index = fmap_width_start_index + prototype_shape[-1]

                # TODO 3Dify
                batch_max_fmap_patch_j = protoL_input_torch[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index,
                ]

                # TODO: CONSTRUCT DICTIONARY OUTSIDE THE LOOP ONCE
                # FIXME: We should enforce sample_id is not None

                global_prototype_meta[j] = PrototypeFromSampleSource(
                    sample_id=sample_ids[img_index_in_batch],
                    target=labels[img_index_in_batch].detach(),
                    # TODO 3Dify
                    latent_patches=latent_prototype_patch_map(
                        protoL_input_torch.shape[-2:],
                        [
                            (fmap_height_start_index, fmap_height_end_index),
                            (fmap_width_start_index, fmap_width_end_index),
                        ],
                        device=self.prototype_tensors.device,
                    ),
                    # calculate these outside for efficiency
                    embedding=batch_max_fmap_patch_j.detach(),
                    img=None,
                    hash=None,
                )

                global_max_proto_act[j] = batch_max_proto_act_j.detach()
                global_max_fmap_patches[j] = batch_max_fmap_patch_j.detach()
                updated_prototype_indices[j] = img_index_in_batch

        return updated_prototype_indices

    def prune_prototypes_by_index(self, prune_indices):
        """
        Remove prototypes from the prototype tensor and information dictionary
        based on indices.

        Args:
            prune_indices - list of prototypes by index to remove from layer
        """
        prune_set = set(prune_indices)
        all_indices = list(range(self.prototype_tensors.shape[0]))
        keep_indices = [i for i in all_indices if i not in prune_set]

        new_prototypes = [
            PrototypeWithMeta(
                prototype_tensor=self.prototype_tensors[i],
                meta=self.prototype_meta[i],
            )
            for i in keep_indices
        ]

        self.set_prototypes(new_prototypes)


class ClassAwarePrototypeLayer(PrototypeLayer):
    def __init__(
        self,
        activation_function: Callable,
        prototype_class_identity: torch.Tensor,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        k_for_topk: int = 1,
        class_specific_project: bool = True,
    ):
        """
        Args:
            num_classes: The number of classes for this task
            activation_function: The activation function to use
            prototype_class_identity: A onehot tensor indicating which prototypes
                correspond to which class. prototypes x classes
            latent_channels: The number of output channels from the backbone
            prototype_dimension: A (proto_h, proto_w) tuple indicating the size of
                the prototypes in the latent space
            episilon_val: A small value to prevent division by zero.
        """
        super(ClassAwarePrototypeLayer, self).__init__(
            num_prototypes=prototype_class_identity.shape[0],
            activation_function=activation_function,
            latent_channels=latent_channels,
            prototype_dimension=prototype_dimension,
            k_for_topk=k_for_topk,
        )
        self.num_classes = prototype_class_identity.shape[1]

        self.with_fa = False

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes

        self.class_specific_project = class_specific_project

        self.register_buffer("prototype_class_identity", prototype_class_identity)

    @property
    def min_sparsity(self):
        return self.num_classes * (
            1 + 1 / (self.latent_spatial_size[0] * self.latent_spatial_size[1])
        )

    def set_prototypes(
        self,
        prototypes: List[PrototypeWithMeta],
    ):
        """
        Sets the prototypes to the given list of prototypes.
        If a prototype is given as a tensor, the metadata is set to a PrototypeRandomInitMeta.

        For the class aware prototype layer, the prototypes can only be updated if they have metadata.
        """
        super(ClassAwarePrototypeLayer, self).set_prototypes(prototypes)
        # Ensure that the prototypes are class-aware
        prototype_class_identity = torch.zeros(
            len(prototypes), self.num_classes, device=self.prototype_tensors.device
        )
        for i, proto in enumerate(prototypes):
            prototype_class_identity[i, proto.meta.target] = 1

        self.prototype_class_identity.data = prototype_class_identity

    def get_extra_state(self):
        extra_state = super(ClassAwarePrototypeLayer, self).get_extra_state()
        extra_state["class_specific_project"] = self.class_specific_project
        return extra_state

    def set_extra_state(self, state):
        super(ClassAwarePrototypeLayer, self).set_extra_state(state)
        self.class_specific_project = state["class_specific_project"]

    def get_protos_for_rsample_init(self, dataset):
        class_ids = torch.argmax(self.prototype_class_identity, axis=1)

        # a dict of {labels:[pos1, pos2,...], ...} where pos are which proto index
        protolabels_to_protoindices = dict()
        for idx, value in enumerate(class_ids.tolist()):
            if value in protolabels_to_protoindices:
                protolabels_to_protoindices[value].append(idx)
            else:
                protolabels_to_protoindices[value] = [idx]

        # which prototype to return
        # this is num prototypes sized
        init_samples = [None for i in range(class_ids.shape[0])]
        labels = [None for i in range(class_ids.shape[0])]  # just for checking
        sample_ids = [None for i in range(class_ids.shape[0])]

        # continue until a sample for every position for every class has been found
        while any(len(indices) > 0 for indices in protolabels_to_protoindices.values()):
            # Select random input
            random_int = random.randint(0, len(dataset) - 1)
            img = dataset[random_int]["img"]
            label = dataset[random_int]["target"]

            # continue if reached max for a specific class
            if len(protolabels_to_protoindices[label]) == 0:
                continue

            # get the correct position of prototype
            position = protolabels_to_protoindices[label].pop(0)

            # place the img in the correct position
            init_samples[position] = img
            labels[position] = label
            sample_ids[position] = dataset[random_int]["sample_id"]

        return init_samples, labels, sample_ids

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        global_prototype_meta: List[PrototypeFromSampleSource],
        sample_ids,
        labels,
    ):
        prototype_layer_stride = 1
        updated_prototype_indices = {}

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device)
        )["prototype_activations"]

        # protoL_input_ = torch.clone(protoL_input_torch.detach().cpu())
        # proto_act_ = torch.clone(proto_act_torch.detach().cpu())

        # del protoL_input_torch, proto_act_torch

        if self.class_specific_project:
            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(labels):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        prototype_shape = self.prototype_tensors.shape

        for j in range(self.num_prototypes):
            class_index = j

            if self.class_specific_project:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    self.prototype_class_identity[class_index]
                ).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_torch[class_to_img_index_dict[target_class]][
                    :, j, :, :
                ]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_act_j = proto_act_torch[:, j, :, :]
            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )
                if self.class_specific_project:
                    """
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    """
                    batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][
                        batch_argmax_proto_act_j[0]
                    ]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                fmap_height_end_index = fmap_height_start_index + prototype_shape[-2]
                fmap_width_end_index = fmap_width_start_index + prototype_shape[-1]

                batch_max_fmap_patch_j = protoL_input_torch[
                    img_index_in_batch,
                    :,
                    fmap_height_start_index:fmap_height_end_index,
                    fmap_width_start_index:fmap_width_end_index,
                ]

                global_prototype_meta[j] = PrototypeFromSampleSource(
                    sample_id=sample_ids[img_index_in_batch],
                    target=labels[img_index_in_batch].detach(),
                    latent_patches=latent_prototype_patch_map(
                        protoL_input_torch.shape[-2:],
                        [
                            (fmap_height_start_index, fmap_height_end_index),
                            (fmap_width_start_index, fmap_width_end_index),
                        ],
                        device=self.prototype_tensors.device,
                    ),
                    embedding=batch_max_fmap_patch_j.detach(),
                    # calculate these outside for efficiency
                    hash=None,
                    img=None,
                )

                global_max_proto_act[j] = batch_max_proto_act_j.detach()
                global_max_fmap_patches[j] = batch_max_fmap_patch_j.detach()
                updated_prototype_indices[j] = img_index_in_batch

        return updated_prototype_indices


class DeformablePrototypeLayer(ClassAwarePrototypeLayer):
    def __init__(
        self,
        prototype_class_identity: torch.Tensor,
        offset_predictor: nn.Module = None,
        latent_channels: int = 512,
        prototype_dimension: tuple = (1, 1),
        epsilon_val=1e-5,
        activation_function=CosPrototypeActivation(),
        prototype_dilation: int = 1,
        class_specific_project: bool = True,
        k_for_topk: int = 1,
    ):
        """
        Args:
            prototype_class_identity: A onehot tensor indicating which prototypes
                correspond to which class
            offset_predictor: A function that takes as input the latent
                tensor x of shape (batch, channel, height, width) and produces
                a tensor of offsets of shape (batch, 2, proto_h, proto_w, height, width)
                (2 is for a x, y offset for each part at each location)
            latent_channels: The number of output channels from the backbone
            prototype_dimension: A (proto_h, proto_w) tuple indicating the size of
                the prototypes
            episilon_val: A small value to prevent division by zero.
            activation_function: The activation function to use
            class_specific_project: Whether projection is restricted to the class of the prototype
            k_for_topk: The number of prototypes to consider for each image
        """
        super(DeformablePrototypeLayer, self).__init__(
            prototype_class_identity=prototype_class_identity,
            latent_channels=latent_channels,
            prototype_dimension=prototype_dimension,
            activation_function=activation_function,
            class_specific_project=class_specific_project,
            k_for_topk=k_for_topk,
        )

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        self.epsilon_val = epsilon_val
        if offset_predictor is None:
            self.offset_predictor = ConvolutionalSharedOffsetPred(
                prototype_shape=self.prototype_tensors.shape,
                input_feature_dim=latent_channels,
                prototype_dilation=prototype_dilation,
            )
        else:
            self.offset_predictor = offset_predictor

    def forward(self, x: torch.Tensor, sample_locs: torch.Tensor = None):
        """
        Args:
            x: The input tensor of shape (batch_size, feature_dim, latent_height, latent_width)
            sample_locs (optional): A (batch, proto_h, proto_w, H, W, 2) tensor indicating the
                locations to compare each prototypical part to at each center location. If not
                provided, computed internally.

        Returns: activations (torch.Tensor): Tensor of the activations. This is of shape (batch_size, num_prototypes, activation_height, activation_width).
        """
        if sample_locs is None:
            # offsets comes out as (batch, proto_h, proto_w, 2, height, width)
            offsets = self.offset_predictor(x)

            # View offsets as (batch, proto_h, proto_w, 2, height, width)
            offsets_reshaped = offsets.view(
                x.shape[0],
                self.prototype_tensors.shape[-2],
                self.prototype_tensors.shape[-1],
                2,
                x.shape[-2],
                x.shape[-1],
            )
            # Move our x,y offset dim to end
            offsets_reshaped = offsets_reshaped.permute(0, 1, 2, 4, 5, 3)

            # Figure out which locations are being sampled in normalized (-1, 1) space
            # Comes out as (batch, proto_h, proto_w, H, W, 2)
            sample_locs = self._offsets_to_sample_locs(offsets_reshaped)

        stacked_interp_x = []
        stacked_proto = []
        for proto_h in range(self.prototype_tensors.shape[-2]):
            for proto_w in range(self.prototype_tensors.shape[-1]):
                stacked_proto.append(self.prototype_tensors[:, :, proto_h, proto_w])
                stacked_interp_x.append(
                    F.grid_sample(
                        x,
                        sample_locs[:, proto_h, proto_w].flip(dims=[-1]),
                        align_corners=True,
                    )
                )

        stacked_interp_x = torch.cat(stacked_interp_x, dim=1)
        stacked_proto = torch.cat(stacked_proto, dim=1).unsqueeze(-1).unsqueeze(-1)

        activations = self.activation_function(stacked_interp_x, stacked_proto)

        if not hasattr(self, "latent_spatial_size") or self.latent_spatial_size is None:
            self.latent_spatial_size = (activations.shape[-2], activations.shape[-1])

        output_dict = {
            "prototype_activations": activations,
            "prototype_sample_location_map": sample_locs,
        }
        return output_dict

    def get_extra_state(self):
        extra_state = super(DeformablePrototypeLayer, self).get_extra_state()
        extra_state["offset_predictor"] = self.offset_predictor
        extra_state["epsilon_val"] = self.epsilon_val
        return extra_state

    def set_extra_state(self, state):
        super(DeformablePrototypeLayer).set_extra_state(state)
        self.offset_predictor = state["offset_predictor"]
        self.epsilon_val = state["epsilon_val"]

    def _offsets_to_sample_locs(self, offsets):
        """
        Convert offsets relative to a center location to absolute coordinates,
        normalized between -1 and 1
        Args:
            offsets: A (batch_size, proto_h, proto_w, height, width, 2) tensor
                of offsets relative to the center location at each (height, width)

        Returns: sample_locs, a (batch, proto_h, proto_w, height, width, 2) tensor
            describing the location to compare each prototypical part to at each
            center location and image in the batch
        """
        # Assumes offsets are in unnormalized space, e.g. an offset of
        # 1 moves us by 1 latent cell
        second_last_dim_inits = (
            torch.arange(offsets.shape[-3], device=offsets.device).view(1, 1, 1, -1, 1)
            * 1.0
        )
        offsets[..., 0] += second_last_dim_inits
        offsets[..., 0] /= max(offsets.shape[-3] - 1, 1)

        last_dim_inits = (
            torch.arange(offsets.shape[-2], device=offsets.device).view(1, 1, 1, 1, -1)
            * 1.0
        )
        offsets[..., 1] += last_dim_inits
        offsets[..., 1] /= max(offsets.shape[-2] - 1, 1)

        # offsets are now positions in (0, 1); map them to (-1, 1)
        # for grid sample
        return (offsets - 0.5) * 2

    def update_prototypes_on_batch(
        self,
        protoL_input_torch,
        start_index_of_search_batch,
        global_max_proto_act,
        global_max_fmap_patches,
        global_prototype_meta: List[PrototypeFromSampleSource],
        sample_ids,
        labels,
    ):
        prototype_layer_stride = 1
        updated_prototype_indices = {}

        # Assuming data is on correct device; setup belongs in the trainer
        # TODO: ALL ON CUDA OR NOT
        proto_act_torch = self.forward(
            protoL_input_torch.to(self.prototype_tensors.device)
        )["prototype_activations"]

        if self.class_specific_project:
            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(self.num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(labels):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

        for j in range(self.num_prototypes):
            class_index = j

            if self.class_specific_project:
                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(
                    self.prototype_class_identity[class_index]
                ).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_torch[class_to_img_index_dict[target_class]][
                    :, j, :, :
                ]
            else:
                # if it is not class specific, then we will search through
                # every example
                proto_act_j = proto_act_torch[:, j, :, :]
            batch_max_proto_act_j = torch.amax(proto_act_j)

            if batch_max_proto_act_j > global_max_proto_act[j]:
                batch_argmax_proto_act_j = list(
                    custom_unravel_index(
                        torch.argmax(proto_act_j, axis=None), proto_act_j.shape
                    )
                )
                if self.class_specific_project:
                    """
                    change the argmin index from the index among
                    images of the target class to the index in the entire search
                    batch
                    """
                    batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][
                        batch_argmax_proto_act_j[0]
                    ]

                # retrieve the corresponding feature map patch
                img_index_in_batch = batch_argmax_proto_act_j[0]
                fmap_height_start_index = (
                    batch_argmax_proto_act_j[1] * prototype_layer_stride
                )
                fmap_width_start_index = (
                    batch_argmax_proto_act_j[2] * prototype_layer_stride
                )

                # TODO: REVISIT SHAPE INDEXING
                # Figure out where to sample prototype from
                # offsets comes out as (batch, proto_h, proto_w, 2, height, width)
                offsets = self.offset_predictor(protoL_input_torch)

                # View offsets as (batch, proto_h, proto_w, 2, height, width)
                offsets_reshaped = offsets.view(
                    protoL_input_torch.shape[0],
                    self.prototype_tensors.shape[-2],
                    self.prototype_tensors.shape[-1],
                    2,
                    protoL_input_torch.shape[-2],
                    protoL_input_torch.shape[-1],
                )
                # Move our x,y offset dim to end
                offsets_reshaped = offsets_reshaped.permute(0, 1, 2, 4, 5, 3)

                # Figure out which locations are being sampled in normalized (-1, 1) space
                # Comes out as (batch, proto_h, proto_w, H, W, 2)
                sample_locs = self._offsets_to_sample_locs(offsets_reshaped)

                batch_max_fmap_patch_j = torch.empty(
                    (
                        protoL_input_torch.shape[1],
                        self.prototype_tensors.shape[-2],
                        self.prototype_tensors.shape[-1],
                    )
                )
                for proto_h in range(self.prototype_tensors.shape[-2]):
                    for proto_w in range(self.prototype_tensors.shape[-1]):
                        resampled_for_part = F.grid_sample(
                            protoL_input_torch,
                            sample_locs[:, proto_h, proto_w].flip(dims=[-1]),
                            align_corners=True,
                        )
                        batch_max_fmap_patch_j[:, proto_h, proto_w] = (
                            resampled_for_part[
                                img_index_in_batch,
                                :,
                                fmap_height_start_index,
                                fmap_width_start_index,
                            ]
                        )
                        del resampled_for_part

                global_prototype_meta[j] = PrototypeFromSampleSource(
                    sample_id=sample_ids[img_index_in_batch],
                    target=labels[img_index_in_batch].detach(),
                    # TODO 3Dify
                    latent_patches=latent_prototype_patch_map_deformed(
                        protoL_input_torch.shape[-2:],
                        sample_locs[
                            img_index_in_batch,
                            :,
                            :,
                            fmap_height_start_index,
                            fmap_width_start_index,
                        ],
                        device=self.prototype_tensors.device,
                    ),
                    # calculate these outside for efficiency
                    embedding=batch_max_fmap_patch_j.detach(),
                    sample_locations=sample_locs[img_index_in_batch],
                    img=None,
                    hash=None,
                )

                global_max_proto_act[j] = batch_max_proto_act_j.detach()
                global_max_fmap_patches[j] = batch_max_fmap_patch_j.detach()

                del sample_locs, offsets_reshaped, offsets, batch_max_fmap_patch_j
                updated_prototype_indices[j] = img_index_in_batch

        return updated_prototype_indices
