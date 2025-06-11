import hashlib
import logging
import time
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .prediction_heads import LinearClassPrototypePredictionHead
from .prototype_layers import PrototypeFromSampleSource, PrototypeWithMeta
from .utilities.project_utilities import hash_func

logger = logging.getLogger(__name__)


class ProtoPNet(nn.Module):
    def __init__(
        self,
        backbone,
        add_on_layers,
        activation,
        prototype_layer,
        prototype_prediction_head,
        warn_on_errors: bool = False,
        k_for_topk: int = 1,
    ):
        super(ProtoPNet, self).__init__()

        self.backbone = backbone
        self.add_on_layers = add_on_layers
        self.activation = activation
        self.prototype_layer = prototype_layer
        self.prototype_prediction_head = prototype_prediction_head

        self._has_projected = False

        self.__validate_model(warn_on_errors)

    def __validate_model(self, warn_on_errors: bool = False):
        """
        Validate the integretity of the model - namely, that the three layers are compatible with each other.
        """

        errors = []

        prototype_layer_latent_channels = self.prototype_layer.latent_channels

        if hasattr(self.add_on_layers, "proto_channels"):
            addon_latent_channels = self.add_on_layers.proto_channels
            if addon_latent_channels != prototype_layer_latent_channels:
                errors.append(
                    f"Backbone latent dimension {addon_latent_channels} does not match prototype layer latent dimension {prototype_layer_latent_channels}"
                )

        if getattr(self.prototype_layer, "update_prototypes_on_batch", None) is None:
            errors.append(
                "Prototype layer does not have a push method. This is required for."
            )

        if len(errors) == 0:
            logger.debug("Model validation passed.")
        elif warn_on_errors:
            for error in errors:
                logger.warning(error)
        else:
            for error in errors:
                logger.error(error)
            raise ValueError(
                f"Model validation failed with {len(errors)}. See log for details."
            )

    def prototypes_embedded(self) -> bool:
        """
        Returns whether the current prototypes are aligned with their sample embeddings.
        """
        return self._has_projected and self._project_backbone_hash == param_hash(
            self.backbone
        )

    def get_prototype_complexity(self, decimal_precision=8):
        """
        Computes and returns metrics about how many unique prototypes,
        unique parts, etc the model has
        Args:
            decimal_precision: The number of decimal places up to which we consider for
                equality. I.e., if decimal_precision = 8, 1e-9 equals 2e-9, but 1e-7 != 2e-7
        """
        return self.prototype_layer.get_prototype_complexity(
            decimal_precision=decimal_precision
        )

    def forward(
        self,
        x: torch.Tensor,
        return_prototype_layer_output_dict: bool = False,
        **kwargs,
    ):
        latent_vectors = self.backbone(x)
        latent_vectors = self.add_on_layers(latent_vectors)

        prototype_layer_output_dict = self.prototype_layer(latent_vectors)

        prototype_similarities = prototype_layer_output_dict["prototype_activations"]

        prediction_logits = self.prototype_prediction_head(
            prototype_similarities, **kwargs
        )

        if return_prototype_layer_output_dict:
            output_dict = prediction_logits.copy()
            output_dict.update(prototype_layer_output_dict.copy())
            return output_dict
        else:
            return prediction_logits

    def embed(self, x: torch.Tensor):
        """
        Embeds the input tensor into the prototype space.
        Args:
            x: The input tensor to embed
        Returns:
            The embedded tensor
        """
        latent_vectors = self.backbone(x)
        latent_vectors = self.add_on_layers(latent_vectors)
        return {"latent_vectors": latent_vectors}

    def prune_prototypes(self):
        """
        If a prototypical part model uses pruning, this method of
        the model is reponsible for defining the logic on 'how' a
        prototype is pruned. I.e. it identifies the indices of
        prototypes to be pruned, and calls helper functions to
        prune them by index.

        Not to be confused with `prune_prototypes_by_index` - who
        is solely responsible for updating the prototypes by a
        given index.

        """
        logger.info("Pruning with default logic: removing duplicates")
        self.prune_duplicate_prototypes()

    def prune_prototypes_by_index(self, list_indices):
        """
        Updates all layers of the architecture that contain prototypes
        based on indices generated within the caller function
        `prune_prototypes`.

        Note: for more complex pruning (such as aggregating weights
        in the prediction head based on prototypes pruned or
        inter-prototype relationships), this method will not suffice.
        This method does not carry additional information on how
        the prototypes should be prune.

        Args:
            list_indices - list of prototype indices that should be
                relative to the prototype tensor contained in the
                prototype layer

        Returns:
        None - The function prunes and reindexes the prototypes in the
            prototype layer and prediction head (matching indices)
        """
        # list of indicies containing the prototypes that need to be pruned in the prediction head and prototype layer
        self.prototype_layer.prune_prototypes_by_index(list_indices)
        self.prototype_prediction_head.prune_prototypes_by_index(list_indices)

    def prune_duplicate_prototypes(self, decimal_precision=8) -> None:
        """
        Note that this does not rely on metadata, but on the actual prototypes.
        If the backbone is very simple, it is possible to embed different images very close to each other, and this can lead to
        prototypes that are embedded duplicates but not pixel space duplicates.
        """
        assert (
            type(self.prototype_prediction_head) is LinearClassPrototypePredictionHead
        ), "Error: Pruning only supports linear last layer at the moment"

        visited_unique_prototypes = None
        visited_prototype_class_identities = None
        visited_prototype_last_layer_weight = None

        updated_prototype_meta = [None] * self.prototype_tensors.shape[0]

        new_ind_for_proto = 0
        for proto_ind in range(self.prototype_tensors.shape[0]):
            cur_proto = self.prototype_tensors[proto_ind].unsqueeze(0)
            if visited_unique_prototypes is None:
                visited_unique_prototypes = cur_proto
                visited_prototype_class_identities = (
                    self.prototype_layer.prototype_class_identity[proto_ind].unsqueeze(
                        0
                    )
                )
                visited_prototype_last_layer_weight = (
                    self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ].unsqueeze(1)
                )

                updated_prototype_meta[
                    new_ind_for_proto
                ] = self.prototype_layer.prototype_meta[proto_ind]
                new_ind_for_proto += 1
            else:
                equiv_protos = (
                    torch.isclose(visited_unique_prototypes, cur_proto)
                    .all(axis=1)
                    .all(axis=1)
                    .all(axis=1)
                )
                if equiv_protos.any():
                    target_equiv_proto = torch.argmax(equiv_protos * 1)
                    visited_prototype_last_layer_weight[
                        :, target_equiv_proto
                    ] += self.prototype_prediction_head.class_connection_layer.weight.data[
                        :, proto_ind
                    ]
                else:
                    visited_unique_prototypes = torch.cat(
                        [visited_unique_prototypes, cur_proto], dim=0
                    )
                    visited_prototype_class_identities = torch.cat(
                        [
                            visited_prototype_class_identities,
                            self.prototype_layer.prototype_class_identity[
                                proto_ind
                            ].unsqueeze(0),
                        ],
                        dim=0,
                    )
                    visited_prototype_last_layer_weight = torch.cat(
                        [
                            visited_prototype_last_layer_weight,
                            self.prototype_prediction_head.class_connection_layer.weight.data[
                                :, proto_ind
                            ].unsqueeze(
                                1
                            ),
                        ],
                        dim=1,
                    )

                    updated_prototype_meta[
                        new_ind_for_proto
                    ] = self.prototype_layer.prototype_meta[proto_ind]
                    new_ind_for_proto += 1

        logger.info(
            f"Pruning from {self.prototype_tensors.shape[0]} prototypes to {visited_unique_prototypes.shape[0]}"
        )

        # note this truncates the prototypes to the new length
        self.prototype_layer.set_prototypes(
            [
                PrototypeWithMeta(
                    prototype_tensor=visited_unique_prototypes[i],
                    meta=updated_prototype_meta[i],
                )
                for i in range(new_ind_for_proto)
            ]
        )

        # TODO - the following lines should be a "reweight" function in the prototype layer
        new_last_layer = torch.nn.Linear(
            visited_unique_prototypes.shape[0],
            self.prototype_layer.num_classes,
            bias=False,
        ).to(self.prototype_layer.prototype_tensors.device)

        new_last_layer.weight.data.copy_(visited_prototype_last_layer_weight)
        self.prototype_prediction_head.class_connection_layer = new_last_layer
        if hasattr(self.prototype_prediction_head, "prototype_class_identity"):
            # questionable if either of these should exist or if this should just be a property of PrototypicalPartModel
            self.prototype_prediction_head.prototype_class_identity.data = (
                self.prototype_layer.prototype_class_identity.data
            )

    def get_extra_state(self):
        state = {
            "has_projected": self._has_projected,
        }
        if self._has_projected:
            state["project_backbone_hash"] = self._project_backbone_hash
        return state

    def set_extra_state(self, state):
        self._has_projected = state["has_projected"]
        if self._has_projected:
            self._project_backbone_hash = state["project_backbone_hash"]

    def rsample_init(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Randomly initializations the prototypes from samples contained in dataloader
        """
        start = time.time()
        logger.info("Initializing the Prototypes to Random Samples")
        try:
            original_mode = self.training
            self.eval()

            # returns a list of imgs to be used to initialization
            (
                samples_to_use,
                labels,
                sample_ids,
            ) = self.prototype_layer.get_protos_for_rsample_init(dataloader.dataset)

            combined_inputs = torch.stack(samples_to_use, dim=0)
            dataset = TensorDataset(combined_inputs)
            init_proto_loader = DataLoader(
                dataset, batch_size=dataloader.batch_size, shuffle=False
            )

            embeddings = []
            for batch in init_proto_loader:
                input = batch[0]
                output = self.add_on_layers(self.backbone(input.cuda()))
                embeddings.append(output)

            embeddings = torch.cat(embeddings, dim=0)
            self.prototype_layer.rsample_init(embeddings, labels, sample_ids)
            end = time.time()
            logger.info(
                "\tRandom Sample Initialization time: \t{0}".format(end - start)
            )
        finally:
            self.training = original_mode

    def project(
        self, dataloader: torch.utils.data.DataLoader, transform: callable = None
    ) -> List[PrototypeFromSampleSource]:
        """
        Args:
            dataloader: DataLoader containing the samples to project onto prototypes

        Returns:
            List of PrototypeSourceSample objects, containing complete information about the projected prototypes.
        """
        logger.info("projecting prototypes onto %s", dataloader)
        state_before_push = self.training
        self.eval()
        start = time.time()

        # TODO: RENAME THIS
        n_prototypes = self.prototype_layer.num_prototypes

        global_max_proto_act = torch.full((n_prototypes,), -float("inf"))
        global_max_fmap_patches = torch.zeros_like(
            self.prototype_layer.prototype_tensors
        )
        global_prototype_source: List[PrototypeFromSampleSource] = list(
            self.prototype_layer.prototype_meta
        )

        search_batch_size = dataloader.batch_size

        logger.debug("initiating project batches")

        for push_iter, batch_data_dict in enumerate(tqdm(dataloader)):
            # TODO: ADD TQDM OPTIONALITY TO THIS LOOP
            logger.debug("starting project batch")

            batch_data_dict["img"] = batch_data_dict["img"].to(
                self.prototype_layer.prototype_tensors.device
            )

            if transform is not None:
                batch_data_dict = transform(batch_data_dict)

            search_batch_input = batch_data_dict["img"]
            labels = batch_data_dict["target"]
            try:
                sample_ids = batch_data_dict["sample_id"]
            except KeyError:
                sample_ids = None

            start_index_of_search_batch = push_iter * search_batch_size

            search_batch_input = search_batch_input.to(
                self.prototype_layer.prototype_tensors.device
            )

            logger.debug("updating current best prototypes")
            batch_updated_prototype_indices = (
                self.prototype_layer.update_prototypes_on_batch(
                    self.add_on_layers(self.backbone(search_batch_input)),
                    start_index_of_search_batch,
                    global_max_proto_act,
                    global_max_fmap_patches,
                    global_prototype_source,
                    sample_ids,
                    labels,
                )
            )
            logger.debug("project batch complete")

            # update the global metadata to include the fields taht are not calculated in the prototype layer
            for (
                proto_index,
                img_index_in_batch,
            ) in batch_updated_prototype_indices.items():
                global_prototype_source[proto_index].img = search_batch_input[
                    img_index_in_batch
                ].detach()

        # calculate hash at the end to avoid intermediate recalcs
        for proto_meta in global_prototype_source:
            proto_meta.hash = hash_func(proto_meta.img)

        prototype_with_meta = []
        for proto_tensor, source in zip(
            global_max_fmap_patches, global_prototype_source
        ):
            prototype_with_meta.append(
                PrototypeWithMeta(prototype_tensor=proto_tensor, meta=source.as_meta())
            )

        self.prototype_layer.set_prototypes(prototype_with_meta)

        end = time.time()
        logger.info("\tpush time: \t{0}".format(end - start))
        self.train(state_before_push)

        self._has_projected = True
        self._project_backbone_hash = param_hash(self.backbone)

        return global_prototype_source

    @property
    def prototype_tensors(self) -> torch.Tensor:
        return self.prototype_layer.prototype_tensors.data

    @property
    def num_prototypes(self) -> int:
        return self.prototype_layer.num_prototypes

    def get_prototype_class_identity(self, label) -> torch.Tensor:
        return self.prototype_layer.prototype_class_identity[:, label]

    def input_channels(self) -> torch.Tensor:
        """
        Returns: The number of input channels to the model
        """
        return self.backbone.input_channels

    def describe_prototypes(self):
        # Resturn string describing the prototypes
        ret_str = ""
        for proto_index, proto_info in enumerate(self.prototype_layer.prototype_meta):
            class_connection_vector = (
                self.prototype_prediction_head.class_connection_layer.weight[
                    :, proto_index
                ]
            )
            closest_class = torch.argmax(class_connection_vector)
            ret_str += f"\nPrototype {proto_index} comes from sample {proto_info.sample_id}.\n\tIt has highest class connection to class {closest_class} with a class connection vector of:\n\t\t{class_connection_vector}"
        return ret_str


def param_hash(layer, precision=8):
    """
    Creates a hash of model parameters with controlled precision.
    """
    param_buffer = []
    scaling = 10**precision

    for param in layer.parameters():
        # Round using torch operations before converting to bytes
        param_rounded = (param.detach().cpu() * scaling).round() / scaling
        # Convert to bytes only at the final step
        param_buffer.append(param_rounded.numpy().tobytes())

    hasher = hashlib.sha256()
    for param_bytes in param_buffer:
        hasher.update(param_bytes)
    return hasher.hexdigest()
