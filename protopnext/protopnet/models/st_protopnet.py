from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from ..model_losses import ProtoPNetLoss
from ..prototype_layers import PrototypeFromSampleSource
from ..prototypical_part_model import ProtoPNet
from ..train.metrics import TrainingMetrics
from ..train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
    ProjectPhase,
    TrainLayersUsingProtoPNetNames,
    TrainOptimizerLayers,
)
from ..train.scheduling.types import (
    IterativePhase,
    Phase,
    PostForwardCalculation,
    SetTrainingLayers,
    TrainingSchedule,
)
from ..utilities.trainer_utilities import init_or_update


class CachedForward:
    """
    Helper class to cache the forward result - when enabled - of the backbone.
    Functions similar to a decorator but supports model saving.

    ST ProtoPNet benefits:
    - Reduce computations of the backbone when performing sequential forwards
        of underlying sub-models (see `STProtoPNet.forward(.)`).
    - reduce memory input of autograd when backbone output is being reused for
        sub-models (again, see `STProtoPNet.forward(.))

    Note:
    - this cache behavior should only be used for sequential sub-model forwards!
        If the cache is still enabled when entering project epoch (or other
        epochs that use the backbone directly), then expect erroneous results.
        Make sure to disable cache if you want each backbone input to be processed.
    """

    def __init__(self, forward_func):
        self.forward_func = forward_func
        self.cache = None
        self.enable_cache = False

    def __call__(self, instance, *args, **kwargs):

        if self.cache is not None and self.enable_cache:
            output = self.cache
        else:
            output = self.forward_func(instance, *args, **kwargs)
            if self.enable_cache:
                self.cache = output

        return output

    def toggle_cache(self, toggle: bool) -> None:
        """
        Toggle cache behavior. If `True`, model will cache layer forward and return
        cached result after sequential forwards. If `False`, model forward will
        be calculated (not re-using cache) and cache will be emptied.

        Args:
            toggle (bool): sets the cache functionality on (`toggle=True`) or off
            (`toggle=False`).

        Returns:
            None: Model forwards will be cached (`toggle=True`) or not (`toggle=False`)

        Notes:
            - Setting the toggle to false also clears the cache

        """
        self.enable_cache = toggle
        if toggle is False:
            self.reset_cache()

    def reset_cache(self) -> None:
        """
        Empty cache output from layer forward.

        Returns:
            None: Cache is emptied.
        """
        self.cache = None


class STProtoPNet(ProtoPNet):
    """
    ST-ProtoPNet is a prototypical part network inspired by the support vector
    machine (https://arxiv.org/abs/2301.04011). The model consistes of two
    banches. Each branch is modeled by a vanilla ProtoPNet and share only the
    backbone layer. The output of this model is a collection of sub-model ouputs.
    For understanding how model classification occurs, how loss is calculated,
    and how the model is trained, please view `STProtoPNetClassificationPhase`
    and `STProtoPNetBackpropPhase` phases.
    """

    def __init__(self, models: List[ProtoPNet], **kwargs):
        """
        Construct a STProtoPNet model from a list of `ProtoPNet` sub-models.
        This class functions as joint model interface meaning that any
        library functionality will call instances of this class as if it were
        also a `ProtoPNet`.

        Note:
        - This model assumes that the backbone is shared and will cache backbone
            latent vectors based on the first `ProtoPNet.backbone.forward(.)` call.
        """
        super(ProtoPNet, self).__init__()

        models[0].backbone.forward = CachedForward(models[0].backbone.forward)
        models[0].add_on_layers.forward = CachedForward(models[0].add_on_layers.forward)

        self._models = torch.nn.ModuleList(models)

        self.reinit_parameters()

    @property
    def models(self) -> List[ProtoPNet]:
        return self._models

    def named_parameters(self, prefix="", recurse=True):
        """
        This method truncates the param names from `torch.nn.Module.named_parameters()`.
        This is need to accomodate the `set_training_layers` method.
        """
        named_params = super().named_parameters(prefix, recurse)
        # skip the module list names
        for name, param in named_params:
            # Yield the modified tuple instead of constructing a list
            yield (".".join(name.split(".")[2:]), param)

    @property
    def backbone(self) -> torch.nn.Module:
        # NOTE: not a general solution - unique to ST ProtoPNet
        return self.models[0].backbone

    @property
    def add_on_layers(self) -> torch.nn.Module:
        return self.models[0].add_on_layers

    def prototypes_embedded(self) -> bool:
        for model in self.models:
            if not model.prototypes_embedded():
                return False
        return True

    def get_prototype_complexity(self, decimal_precision=8):
        support_complexity = self.models[0].get_prototype_complexity(decimal_precision)
        trivial_complexity = self.models[1].get_prototype_complexity(decimal_precision)

        return {
            "n_unique_proto_parts": support_complexity['n_unique_proto_parts'] + trivial_complexity['n_unique_proto_parts'],
            "n_unique_protos": support_complexity['n_unique_protos'] + trivial_complexity['n_unique_protos'],
            # Note this assumes the same number of prototypes for each model
            "prototype_sparsity": (support_complexity['prototype_sparsity'] + trivial_complexity['prototype_sparsity'])/2,
        }

    def forward(
        self,
        x: torch.Tensor,
        combine_results: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        This method defines how the forward results of the `ProtoPNet` sub-models
        are combined.

        Args:
            x (torch.Tensor): Model forward input.
            additional_forward_requirements (List[Dict[str, Any]]): List of key
                words defined prior to model forward that may be consumed further
                down the call stack and results captured in the model output
                dictionary.

        Returns:
            List[Dict[str, Any]]: List of model output dictionaries, each contains:
                - sub-model prediction head logits.

                - Additional optional data determined by
                    additional_forward_requirements argument.

        Notes:
            - No kwargs are accepted because sub-model kwarg assignment is needed
                and handled through the additional_forward_requirements list
        """
        # cache: enabled (resume previous active state)
        self.models[0].backbone.forward.toggle_cache(True)
        self.models[0].add_on_layers.forward.toggle_cache(True)

        # sequential model forwards with outputs stored in list
        results = []
        for i, model in enumerate(self.models):
            results.append(model(x, **kwargs))
        # backbone cache: reset (state: emptied), disabled (freeze state)
        self.models[0].backbone.forward.reset_cache()
        self.models[0].backbone.forward.toggle_cache(False)
        self.models[0].add_on_layers.forward.reset_cache()
        self.models[0].add_on_layers.forward.toggle_cache(False)
        if combine_results:
            combined_results = {}
            support_results = results[0]
            trivial_results = results[1]

            combined_results['logits'] = support_results['logits'] + trivial_results['logits']
            combined_results['prototype_activations'] = torch.cat([support_results['prototype_activations'], trivial_results['prototype_activations']], 1)

            return combined_results
        else:
            return results

    def project(
        self, dataloader: torch.utils.data.DataLoader, transform: callable = None
    ) -> List[PrototypeFromSampleSource]:
        results = []
        for model in self.models:
            results.extend(model.project(dataloader, transform))
        return results

    @property
    def num_prototypes(self) -> int:
        return sum([model.prototype_layer.num_prototypes for model in self.models])

    def reinit_parameters(self):
        """
        Based on original code (https://github.com/cwangrun/ST-ProtoPNet/blob/master/full/model.py#L219-L242),
        reinitialize each sub-model's add-on layer weights.
        """
        for m in self.models[0].add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class STProtoPNetTrainingSchedule(TrainingSchedule):
    """
    Scheduler for the STProtoPNet model. This class is responsible for applying
    learning hyperparameters, initializing optimizers, initializing schedulers,
    and managing training phases.
    """

    def __init__(
        self,
        *,
        model: ProtoPNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        project_loader: torch.utils.data.DataLoader,
        support_protopnet_loss: torch.nn.Module,
        trivial_protopnet_loss: torch.nn.Module,
        num_accumulation_batches: int = 1,
        add_on_lr: float = 0.003,
        num_warm_epochs: int = 10,
        num_preproject_joint_epochs: int = 10,
        backbone_lr: float = 1e-4,
        prototype_lr: float = 0.003,
        pred_head_lr: float = 1e-4,
        weight_decay: float = 1e-3,
        joint_lr_step_size: int = 5,
        joint_lr_step_gamma: float = 0.2,
        last_only_epochs_per_project: int = 20,
        joint_epochs_per_phase: int = 10,
        post_project_phases: int = 12,
        phase_config_kwargs: Dict[str, Any] = {},
        extra_training_metrics: Optional[TrainingMetrics] = None,
    ):

        # eval phases calculate loss functions for both support and trivial
        # sub-models. however, eval accuracy is from overal model predictions
        eval = STProtoPNetClassificationPhase(
            name="eval",
            dataloader=val_loader,
            support_loss=support_protopnet_loss,
            trivial_loss=trivial_protopnet_loss,
            extra_training_metrics=extra_training_metrics,
            **phase_config_kwargs,
        )

        backprop_shared_config = {
            "dataloader": train_loader,
            "num_accumulation_batches": num_accumulation_batches,
            "support_loss": support_protopnet_loss,
            "trivial_loss": trivial_protopnet_loss,
            "extra_training_metrics": extra_training_metrics,
            **phase_config_kwargs,
        }

        opt_conf_backbone = {
            "params": model.backbone.parameters(),
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        }

        opt_conf_add_on_layers = {
            "params": model.add_on_layers.parameters(),
            "lr": add_on_lr,
            "weight_decay": weight_decay,
        }

        proto_params = []
        head_params = []

        # pass sub-model parameters by each layer together
        for sub_model in model.models:

            proto_params.append(
                {
                    "params": sub_model.prototype_layer.prototype_tensors,
                    "lr": prototype_lr,
                }
            )

            head_params.append(
                {
                    # FIXME - shouldn't this be model.prototype_prediction_head.parameters()?
                    "params": sub_model.prototype_prediction_head.class_connection_layer.parameters(),
                    "lr": pred_head_lr,
                }
            )

        # during warmup, only train add-on and prototype layers
        warm = STProtoPNetBackpropPhase(
            name="warm",
            optimizer=torch.optim.Adam([opt_conf_add_on_layers] + proto_params),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=False,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
            ),
            **backprop_shared_config,
        )

        joint_optimizer = torch.optim.Adam(
            [opt_conf_backbone, opt_conf_add_on_layers] + proto_params
        )

        # during joint, train all layers but prediction head
        joint = STProtoPNetBackpropPhase(
            name="joint",
            optimizer=joint_optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(
                joint_optimizer,
                step_size=joint_lr_step_size,
                gamma=joint_lr_step_gamma,
            ),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=True,
                train_add_on_layers=True,
                train_prototype_layer=True,
                train_prototype_prediction_head=False,
            ),
            **backprop_shared_config,
        )

        # prediction head layer only training phase
        last_only_shared_config = backprop_shared_config.copy()
        last_only_shared_config["support_loss"] = support_protopnet_loss
        last_only_shared_config["trivial_loss"] = trivial_protopnet_loss
        last_only = STProtoPNetBackpropPhase(
            name="last_only",
            optimizer=torch.optim.Adam(head_params),
            set_training_layers_fn=TrainLayersUsingProtoPNetNames(
                train_backbone=False,
                train_add_on_layers=False,
                train_prototype_layer=False,
                train_prototype_prediction_head=True,
            ),
            **last_only_shared_config,
        )

        # dataloader used to project prototypes for both sub-models
        project = ProjectPhase(
            dataloader=project_loader,
        )

        # additional phases to assist in model convergence
        maybe_iterative_phase = (
            [
                IterativePhase(
                    phases=[
                        Phase(train=last_only, duration=last_only_epochs_per_project),
                        Phase(train=joint, duration=joint_epochs_per_phase),
                        Phase(train=project),
                    ],
                    iterations=post_project_phases,
                )
            ]
            if post_project_phases > 0
            else []
        )

        super().__init__(
            default_eval_phase=eval,
            phases=[
                Phase(train=warm, duration=num_warm_epochs),
                Phase(train=joint, duration=num_preproject_joint_epochs),
                Phase(train=project),
                *maybe_iterative_phase,
                Phase(train=last_only, duration=last_only_epochs_per_project),
            ],
        )


class STProtoPNetClassificationPhase(ClassificationInferencePhase):
    """
    Classification phase specific to ST-ProtoPNet. Since this model uniquely
    provides sub-model forward outputs while alternating branch, this phase uniquely defines how to
    consume joint model output
    """

    def __init__(
        self,
        name,
        dataloader,
        device,
        support_loss: ProtoPNetLoss,
        trivial_loss: ProtoPNetLoss,
        extra_training_metrics=None,
        post_forward_calculations=None,
    ):

        super().__init__(
            name,
            dataloader,
            device,
            support_loss,
            extra_training_metrics,
            post_forward_calculations,
        )

        self._support_loss = support_loss
        self._trivial_loss = trivial_loss

        # Get the unique required forward results from both losses.
        # NOTE: this only works fine so long as we expect the required results not to be overwritten
        unique_req_results = set(
            self._support_loss.batch_loss.required_forward_results()
        ).union(self._trivial_loss.batch_loss.required_forward_results())

        # Create a single forward_calc_flags dict.
        self.forward_calc_flags = {f"return_{req}": True for req in unique_req_results}
        self.forward_calc_flags["return_prototype_layer_output_dict"] = True

        # FIXME: a cleaner fix would be to pass the current batch idx to run_classification
        self._batch_num = 0

        # FIXME: doing this just to avoid difficult to catch bugs
        del self._loss

    @property
    def support_loss(self) -> ProtoPNetLoss:
        return self._support_loss

    @property
    def trivial_loss(self) -> ProtoPNetLoss:
        return self._trivial_loss

    def _calculate_final_metrics(
        self,
        epoch_metrics_dict: Dict[str, Any],
        lr_metrics: Dict[str, float],
        extra_training_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Calculate ST-ProtoPNet phase and epoch level metrics. Since this model
        contains 2 branches/sub-models, we specifically track the number of
        batches each branch was trained on to calculate the correct metric
        average across the epochs.
        """

        maybe_extra_training_metrics = (
            extra_training_metrics if extra_training_metrics is not None else {}
        )

        final_metrics = {**lr_metrics, **maybe_extra_training_metrics}

        batch_averaged_metrics = [
            "total_loss",
            "cross_entropy",
            "nll_loss",
            "cluster",
            "separation",
            "fine_annotation",
            "accu",
            "l1",
            "discrimination_loss",
            "closeness_loss",
            "orthogonality_loss",
        ]

        for k, v in epoch_metrics_dict.items():
            if v is not None:
                if isinstance(v, torchmetrics.metric.Metric):
                    value = v.compute()
                else:
                    value = v

                if k in [
                    "n_examples",
                    "n_correct",
                    "n_batches",
                    "support/n_batches",
                    "trivial/n_batches",
                ]:
                    value = int(value)

                # for metrics we plan to average across all batches of epoch
                if any([metric in k for metric in batch_averaged_metrics]):
                    # by default, use total batches in phase
                    n_batches_key = "n_batches"
                    if "support/" in k:
                        n_batches_key = "support/" + n_batches_key
                    elif "trivial/" in k:
                        n_batches_key = "trivial/" + n_batches_key
                    value = float(value)
                    value = value / float(epoch_metrics_dict[n_batches_key])

                final_metrics[k] = value

        final_metrics["accuracy"] = final_metrics["accu"]

        return final_metrics

    def _select_model_idx(self, batch_num: int, num_accumulation_batches: int) -> int:
        """
        Partion batches into num_accumulation_batches groups and select sub-model
        indices by alternating between groups of batches.

        Note:
            - ST-ProtoPNet assumes only 2 models, hence the modulus is done with 2.

        Example:
        Here is how the batches are grouped:
        >>> batch_indices
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> batch_indices // num_accumulation_batches
        [0, 0, 0, 1, 1, 1, 2, 2, 2] # assuming batch groups of size 3
        >>> batch_indices // num_accumulation_batches % 2
        [0, 0, 0, 1, 1, 1, 0, 0, 0] # idx model assignment with 2 models
        """
        return (batch_num // num_accumulation_batches) % 2

    def run_classification(
        self,
        model: STProtoPNet,
        batch_data_dict: Dict[str, Any],
        epoch_metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run prediction on a batch of data
        """

        input = batch_data_dict["img"]
        target = batch_data_dict["target"]
        output = model(input, combine_results=False, **self.forward_calc_flags)
        # extra sub-model logits (1) for final predictions and (2) model specific
        # loss calculations
        support_output, trivial_output = output[0], output[1]

        with torch.no_grad():
            support_logits = support_output["logits"]
            trivial_logits = trivial_output["logits"]

            # evaluation statistics
            _, predicted = torch.max(support_logits.data + trivial_logits.data, 1)

            init_or_update(epoch_metrics_dict, "n_examples", target.size(0))
            init_or_update(
                epoch_metrics_dict, "n_correct", (predicted == target).sum().item()
            )

        # FIXME: when there is no optimizer (validation set), we should use overall model loss

        num_accumulation_batches = (
            self.num_accumulation_batches
            if hasattr(self, "num_accumulation_batches")
            else 1
        )

        # When training, use alternating loss function
        select_idx = self._select_model_idx(self._batch_num, num_accumulation_batches)
        sub_model = model.models[select_idx]

        complete_output = self.append_extra_forward_results(
            sub_model, batch_data_dict, output[select_idx], "train"
        )

        loss_fn, sub_model_name = (
            (self.trivial_loss, "trivial/")
            if select_idx == 0
            else (self.support_loss, "support/")
        )
        loss, loss_term_dict = loss_fn(
            target=target,
            model=sub_model,
            metrics_dict=epoch_metrics_dict,
            **complete_output,
        )

        for loss_term_item, value in loss_term_dict.items():
            init_or_update(epoch_metrics_dict, sub_model_name + loss_term_item, value)

        # at this point we have 'trivial/loss.name' ...
        init_or_update(epoch_metrics_dict, sub_model_name + "n_batches", 1)

        # do not prepend sub_model since this was calculated with the joint model
        init_or_update(
            epoch_metrics_dict,
            "accu",
            float(epoch_metrics_dict["n_correct"])
            / float(epoch_metrics_dict["n_examples"]),
        )

        self._batch_num += 1

        # FIXME: should we be passing the joint output here?
        return output, loss


class STProtoPNetBackpropPhase(
    STProtoPNetClassificationPhase, ClassificationBackpropPhase
):
    """
    Backprop phase that enables the optimizer during the classification phase.
    Specific to ST-ProtoPNet, this phase will only update one branch/sub-model's
    parameters at a time.

    Note:
        - During each epoch, one branch/sub-model (the ProtoPNet models
            corresponding to the either the support or trivial prototypes) will
            have their gradients required.

        - When a branch is being trained, we use the loss terms corresponding
            to the prototypes in that branch. I.e. support_loss is used for the
            branch containing the support prototypes and trivial_loss is used
            for the branch containing trivial prototypes.
            (Analog from original ST-ProtoPNet code:
            https://github.com/cwangrun/ST-ProtoPNet/blob/master/full/train_and_test.py#L132-L149)

        - In general, accuracy is purely determined by the branches joint output.
            I.e. the average of logits generated by both branches are the final
            logits of the model.
            (Analog from original ST-ProtoPNet code:
            https://github.com/cwangrun/ST-ProtoPNet/blob/master/full/train_and_test.py#L118-L121)
    """

    def __init__(
        self,
        name: str,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        support_loss: ProtoPNetLoss,
        trivial_loss: ProtoPNetLoss,
        optimizer: torch.optim.Optimizer,
        extra_training_metrics: Optional[TrainingMetrics] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        set_training_layers_fn: SetTrainingLayers = TrainOptimizerLayers(),
        num_accumulation_batches: int = 1,
        post_forward_calculations: Optional[Tuple[PostForwardCalculation]] = None,
    ):
        STProtoPNetClassificationPhase.__init__(
            self,
            name,
            dataloader,
            device,
            support_loss,
            trivial_loss,
            extra_training_metrics,
            post_forward_calculations,
        )

        # FIXME: passing the first loss function is probably not a good idea
        ClassificationBackpropPhase.__init__(
            self,
            name,
            dataloader,
            device,
            support_loss,
            optimizer,
            extra_training_metrics,
            scheduler,
            set_training_layers_fn,
            num_accumulation_batches,
            post_forward_calculations,
        )

        # FIXME: doing this just to avoid difficult to catch bugs
        del self._loss

    def _calculate_final_metrics(
        self, epoch_metrics_dict, lr_metrics, extra_training_metrics=None
    ):
        # because of the MRO, we have to triple override this method. without
        # this method, metrics will not be correctly tracked.
        # (https://docs.python.org/3/howto/mro.html)
        return STProtoPNetClassificationPhase._calculate_final_metrics(
            self, epoch_metrics_dict, lr_metrics, extra_training_metrics
        )

    def _pre_metrics_update(self, model, batch_data_dict, output, loss):
        """
        Normalize prototype tensors after backwards.
        """
        # NOTE: this phase is backprop, so it may be safe to assume there
        # is an optimizer and that self.num_accumulation_batches was defined
        if (
            hasattr(self, "optimizer")
            and (self._batch_num % self.num_accumulation_batches) == 0
        ):
            for sub_model in model.models:
                sub_model.prototype_layer.prototype_tensors.data = F.normalize(
                    sub_model.prototype_layer.prototype_tensors, p=2, dim=1
                ).data

    def _set_sub_model_required_gradient(
        self, sub_model: ProtoPNet, required_grad: bool
    ):
        layers = [
            sub_model.prototype_layer,
            sub_model.prototype_prediction_head,
        ]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = required_grad

    def run_classification(self, model, batch_data_dict, epoch_metrics_dict):

        select_idx = self._select_model_idx(
            self._batch_num, self.num_accumulation_batches
        )

        sub_model = model.models[select_idx]

        # NOTE: should be safe to assume there is an optimizer...
        if (
            hasattr(self, "optimizer")
            and (self._batch_num % self.num_accumulation_batches) == 0
        ):
            self._set_sub_model_required_gradient(sub_model, required_grad=True)
            self._set_sub_model_required_gradient(
                model.models[(select_idx + 1) % 2], required_grad=False
            )

        return super().run_classification(model, batch_data_dict, epoch_metrics_dict)
