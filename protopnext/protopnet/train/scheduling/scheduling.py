import inspect
import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple, Union

import torch
import torchmetrics
from tqdm.auto import tqdm

from protopnet.train.checkpointing import ModelCheckpointer
from protopnet.train.logging.types import TrainLogger

from ...model_losses import ProtoPNetLoss
from ...prototypical_part_model import ProtoPNet
from ...utilities.trainer_utilities import get_learning_rates, init_or_update
from ..metrics import TrainingMetrics
from .types import (
    PostForwardCalculation,
    PostPhaseSummary,
    SetTrainingLayers,
    StepContext,
)

log = logging.getLogger(__name__)


class _NoGradPhaseMixin:
    @contextmanager
    def phase_settings(self, model: ProtoPNet):
        original_training_state = model.training
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            model.train(original_training_state)


@dataclass(frozen=True)
class ProjectPhase(_NoGradPhaseMixin):
    dataloader: torch.utils.data.DataLoader
    name: str = "project"

    def run_step(
        self, model: ProtoPNet, step_context: StepContext
    ) -> Optional[Dict[str, Any]]:
        with torch.no_grad():
            model.project(self.dataloader)
        return model.get_prototype_complexity()

    def after_training(
        self,
        model: ProtoPNet,
        metric_logger: TrainLogger,
        checkpointer: ModelCheckpointer,
        post_phase_summary: PostPhaseSummary,
    ) -> Optional[Dict[str, Any]]:
        """
        Clean up at the end of the phase.

        Args:
            model: The model being trained

        Returns:
            Optional dictionary of cleanup results
        """
        metric = post_phase_summary.final_target_metric
        # FIXME - `+ 1` because the previous step is being closed before the post-phase calculations are being run
        metric_logger.log_metrics(
            "project",
            step=post_phase_summary.last_step + 1,
            prototypes_embedded_state=True,
            precalculated_metrics={metric[0]: metric[1]},
        )


@dataclass(frozen=True)
class PrunePrototypesPhase(_NoGradPhaseMixin):
    name: str = "prune_prototypes"

    def run_step(
        self, model: ProtoPNet, step_context: StepContext
    ) -> Optional[Dict[str, Any]]:
        model.prune_prototypes()
        return model.get_prototype_complexity()


@dataclass(frozen=True)
class RSampleInitPhase(_NoGradPhaseMixin):
    dataloader: torch.utils.data.DataLoader
    name: str = "rsample_init"

    def run_step(
        self, model: ProtoPNet, step_context: StepContext
    ) -> Optional[Dict[str, Any]]:
        model.rsample_init(self.dataloader)


class TrainOptimizerLayers:
    def set_training_layers(
        self, model: ProtoPNet, phase: "ClassificationBackpropPhase"
    ):
        """
        Default implementation of the SetTrainingLayers protocol that sets all layers that have an optimizer to be trainable
        and otherwise does not set any layers to be trainable.
        """
        optimizer = phase.optimizer
        for name, param in model.named_parameters():
            if (
                optimizer.param_groups[0]["params"] is None
                or name in optimizer.param_groups[0]["params"]
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False


@dataclass(frozen=True)
class TrainLayersUsingProtoPNetNames:
    """
    Implementation of SetTrainingLayers that sets layers to be trainable based on the names of the layers in the ProtoPNet model.
    DEPRECATED: This is provided only for backwards compatibility and will be removed in the near future.
    TrainOptimizerLayers will handle most use cases and avoids potential issues with the naming of layers.
    """

    train_backbone: bool
    train_add_on_layers: bool
    train_prototype_layer: bool
    train_prototype_prediction_head: bool
    train_conv_offset: bool = False

    def __post_init__(self):
        warnings.warn(
            "This is provided only for backwards compatibility and will be removed in the near future.",
            DeprecationWarning,
            stacklevel=2,
        )

    def overwrite_grad_epoch_settings(self, name, param, setting_attr, should_train):
        pass

    def set_training_layers(
        self, model: ProtoPNet, phase: "ClassificationBackpropPhase"
    ):
        # Map model components to training settings dynamically
        for name, param in model.named_parameters():
            # Extract the component name from the parameter name
            # Assuming the naming convention follows the pattern "<component_name>_..."
            component_name = name.split(".")[0]  # Get the first part of the name

            # Construct the setting attribute name dynamically
            setting_attr = f"train_{component_name}"

            # Check if the corresponding setting attribute exists in the epoch settings
            assert hasattr(
                self, setting_attr
            ), f"Attribute '{setting_attr}' not found in epoch_settings"

            # Since the attribute exists, use getattr to fetch its value
            # The third argument in getattr is not needed anymore since we're asserting the attribute's existence
            should_train = getattr(self, setting_attr)

            # Update the requires_grad based on the setting
            param.requires_grad = should_train

            self.overwrite_grad_epoch_settings(name, param, setting_attr, should_train)


class ClassificationInferencePhase(_NoGradPhaseMixin):
    def __init__(
        self,
        name: str,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        loss: ProtoPNetLoss,
        extra_training_metrics: Optional[TrainingMetrics] = None,
        post_forward_calculations: Optional[Tuple[PostForwardCalculation]] = None,
    ):
        self._name = name
        self._dataloader = dataloader
        self._device = device
        self._extra_training_metrics = extra_training_metrics
        self._loss = loss
        self._post_forward_calculations = post_forward_calculations or tuple()

        # Post-initialization setup
        self._sample_count_metric = torchmetrics.SumMetric()
        self.forward_calc_flags = {
            f"return_{req}": True
            for req in self._loss.batch_loss.required_forward_results()
        }
        self.forward_calc_flags["return_prototype_layer_output_dict"] = True

        self._post_forward_calculations = (
            tuple(self._post_forward_calculations)
            if self._post_forward_calculations
            else tuple()
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        return self._dataloader

    @property
    def extra_training_metrics(self) -> TrainingMetrics:
        return self._extra_training_metrics

    @property
    def loss(self) -> ProtoPNetLoss:
        return self._loss

    @property
    def post_forward_calculations(self) -> Tuple[PostForwardCalculation]:
        return self._post_forward_calculations

    @property
    def device(self) -> torch.device:
        return self._device

    def append_extra_forward_results(
        self,
        model: ProtoPNet,
        input: Dict[str, Any],
        output: Dict[str, Any],
        phase: str,
    ):
        """
        Appends results from `extra_calculations` to the output dictionary based on the current phase.

        Parameters:
            input (Dict[str, Any]): Input data for calculations.
            output (Dict[str, Any]): Existing output data to be updated.
            phase (str): Current phase, used to determine eligible calculations.

        Returns:
            Dict[str, Any]: Updated output dictionary with additional results.

        Raises:
            ValueError: If a calculation's `phase` attribute is invalid or required parameters are missing.

        Notes:
            - Calculations may specify a `phase` (string, set, or callable) to restrict execution.
            - Only available parameters, as determined by the calculation's signature, are passed.
            - Calculation results update `output` directly or are added under a derived key.
            - This will be a part of the BackpropEpoch class in the future.
        """
        output = output.copy()

        for calculation in self.post_forward_calculations:
            # Check phase compatibility
            should_run = True
            # Get phase information if it exists
            if hasattr(calculation, "phase"):
                phase_info = calculation.phase
                # If it's callable, call it
                if callable(phase_info):
                    phase_info = phase_info()

                # Handle the result
                if isinstance(phase_info, str):
                    should_run = phase == phase_info
                elif isinstance(phase_info, Set):
                    should_run = phase in phase_info
                else:
                    raise ValueError(
                        f"Phase specification for {calculation.__class__.__name__} must be "
                        f"a string, set, or callable returning one of those, not {type(phase_info)}"
                    )

            if not should_run:
                continue

            sig = inspect.signature(calculation.__call__)

            super_input = {"model": model, **input, **output}

            # Get required parameters (those without defaults)
            required_params = {
                name
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty
                and param.kind
                not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            }

            # Check if all required parameters are available in kwargs
            missing_params = required_params - super_input.keys()
            if missing_params:
                raise ValueError(
                    f"Missing required parameters for {calculation.__class__.__name__}: {missing_params}"
                )

            # Filter kwargs to only include parameters that exist in the signature
            calculation_args = {
                name: super_input[name]
                for name in sig.parameters
                if name in super_input
            }

            result = calculation(**calculation_args)

            if isinstance(result, dict):
                output.update(result)
            else:
                # Get name from the calculation object or its class
                result_name = (
                    getattr(calculation, "name", None)
                    or getattr(  # Try object attribute first
                        calculation, "__name__", None
                    )
                    or calculation.__class__.__name__.lower()  # Try callable name  # Fallback to class name
                )
                output[result_name] = result

        return output

    def run_classification(
        self,
        model: ProtoPNet,
        batch_data_dict: Dict[str, Any],
        epoch_metrics_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run prediction on a batch of data
        """
        input = batch_data_dict["img"]
        target = batch_data_dict["target"]
        output = model(input, **self.forward_calc_flags)

        with torch.no_grad():
            logits = output["logits"]

            # evaluation statistics
            _, predicted = torch.max(logits.data, 1)

            init_or_update(epoch_metrics_dict, "n_examples", target.size(0))
            init_or_update(
                epoch_metrics_dict, "n_correct", (predicted == target).sum().item()
            )

        complete_output = self.append_extra_forward_results(
            model, batch_data_dict, output, "train"
        )

        loss, loss_term_dict = self.loss(
            model=model,
            **complete_output,
            **batch_data_dict,
        )

        # TODO: revisit how we want to track batch level loss terms
        for loss_term_item, value in loss_term_dict.items():
            init_or_update(epoch_metrics_dict, loss_term_item, value)

        init_or_update(
            epoch_metrics_dict,
            "accu",
            float(epoch_metrics_dict["n_correct"])
            / float(epoch_metrics_dict["n_examples"]),
        )

        return output, loss

    def _pre_dataloader_step_init(self, model: ProtoPNet, step_context: StepContext):
        """
        Initialize the optimizer for the current batch
        """
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad()

    def _handle_loss_and_optimization(self, loss: torch.Tensor, batch_idx: int):
        """Handle loss backward and optimization steps."""
        if hasattr(self, "optimizer"):
            loss.backward(retain_graph=True)

            # Check if we have reached our accumulation threshold
            if ((batch_idx + 1) % self.num_accumulation_batches == 0) or (
                batch_idx + 1 == len(self.dataloader)
            ):
                if hasattr(self, "optimizer"):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

    def _pre_metrics_update(
        self,
        model: ProtoPNet,
        batch_data_dict: Dict[str, Any],
        output: Dict[str, Any],
        loss: torch.Tensor,
    ):
        pass

    def _calculate_final_metrics(
        self,
        epoch_metrics_dict: Dict[str, Any],
        lr_metrics: Dict[str, float],
        extra_training_metrics: Optional[Dict[str, Any]] = None,
    ):
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
        ]

        for k, v in epoch_metrics_dict.items():
            if v is not None:
                if isinstance(v, torchmetrics.metric.Metric):
                    value = v.compute()
                else:
                    value = v

                if k in ["n_examples", "n_correct", "n_batches"]:
                    value = int(value)

                # for metrics we plan to average across all batches of epoch
                if any([metric in k for metric in batch_averaged_metrics]):
                    value = float(value)
                    value = value / float(epoch_metrics_dict["n_batches"])

                final_metrics[k] = value

        final_metrics["accuracy"] = final_metrics["accu"]

        return final_metrics

    def run_step(self, model: ProtoPNet, step_context: StepContext) -> Dict[str, Any]:
        # FIXME: only allow specific metrics to be calculated every epoch without need for condition
        compute_metrics_this_epoch = self.extra_training_metrics is not None

        num_accumulation_batches = (
            self.num_accumulation_batches
            if hasattr(self, "num_accumulation_batches")
            else 1
        )

        # TODO: Add a way to track variables as None/NA if they aren't used
        # Or make this more flexible to be more flexible in the terms it tracks
        epoch_metrics_dict = {
            "time": None,
            "n_examples": None,
            "n_correct": None,
            "n_batches": None,
            "cross_entropy": None,
            "nll_loss": None,
            "cluster": None,
            "separation": None,
            "fine_annotation": None,
            "accu": None,
            "l1": None,
            "total_loss": None,
            "n_unique_proto_parts": None,
            "n_unique_protos": None,
            "prototype_non_sparsity": None,
            "orthogonality_loss": None,
            "grassmannian_orthogonality_loss": None,
            "closeness_loss": None,
            "discrimination_loss": None,
            "contrastive_masked": None,
            "contrastive_unmasked": None,
        }

        if self.extra_training_metrics is not None and compute_metrics_this_epoch:
            self.extra_training_metrics.start_epoch(phase=self.name)

        # the optimizer existing is a proxy for knowing we are training
        if hasattr(self, "optimizer"):
            lr_metrics = get_learning_rates(
                optimizer=self.optimizer, model=model, detailed=False
            )

        else:
            lr_metrics = {}

        self._pre_dataloader_step_init(model=model, step_context=step_context)

        for i, batch_data_dict in tqdm(enumerate(self.dataloader)):
            for key, maybe_tensor in batch_data_dict.items():
                if (
                    isinstance(maybe_tensor, torch.Tensor)
                    and maybe_tensor.device != self.device
                ):
                    batch_data_dict[key] = maybe_tensor.to(self.device)

            output, batch_loss = self.run_classification(
                model=model,
                batch_data_dict=batch_data_dict,
                epoch_metrics_dict=epoch_metrics_dict,
            )

            loss = batch_loss / num_accumulation_batches

            if torch.isnan(loss).any():
                raise ValueError("Model forward produced a Loss with NaN values!")

            init_or_update(epoch_metrics_dict, "n_batches", 1)
            init_or_update(epoch_metrics_dict, "total_loss", batch_loss.item())

            self._handle_loss_and_optimization(loss=loss, batch_idx=i)

            self._pre_metrics_update(
                model=model, batch_data_dict=batch_data_dict, output=output, loss=loss
            )

            if self.extra_training_metrics is not None and compute_metrics_this_epoch:
                log.debug("updating extra metrics")

                # FIXME: somewhere these tensors are being moved off of gpu
                for key, maybe_tensor in batch_data_dict.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        batch_data_dict[key] = maybe_tensor.to(self.device)
                for key, maybe_tensor in output.items():
                    if (
                        isinstance(maybe_tensor, torch.Tensor)
                        and maybe_tensor.device != self.device
                    ):
                        output[key] = maybe_tensor.to(self.device)

                with torch.no_grad():
                    self.extra_training_metrics.update_all(
                        batch_data_dict, output, phase=self.name
                    )

                log.debug("Extra metrics updated")

            del batch_data_dict, output, loss, batch_loss

        if hasattr(self, "scheduler") and self.scheduler is not None:
            # Step scheduler if possible
            self.scheduler.step()

        if self.extra_training_metrics is not None and compute_metrics_this_epoch:
            log.debug("Computing extra metrics")
            start = time.time()
            with torch.no_grad():
                extra_training_metrics = self.extra_training_metrics.compute_dict()
            log.info("Extra metrics calculated in %s", time.time() - start)
        else:
            extra_training_metrics = None

        # Get list metrics (class_aurocs and conf_mat)
        if extra_training_metrics:
            if "weighted_auroc" in extra_training_metrics:
                epoch_metrics_dict["weighted_auroc"] = float(
                    extra_training_metrics["weighted_auroc"]
                )

        if self.extra_training_metrics is not None:
            self.extra_training_metrics.end_epoch(phase=self.name)

        # FIXME - this is a hack to allow torchmetrics that were never initialized to be skipped
        # This will be replaced with a more systematic metrics calculation when metrics are refactored

        final_metrics = self._calculate_final_metrics(
            epoch_metrics_dict,
            lr_metrics,
            extra_training_metrics=extra_training_metrics,
        )

        return final_metrics


class ClassificationBackpropPhase(ClassificationInferencePhase):
    def __init__(
        self,
        name: str,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        loss: ProtoPNetLoss,
        optimizer: torch.optim.Optimizer,
        extra_training_metrics: Optional[TrainingMetrics] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        set_training_layers_fn: SetTrainingLayers = TrainOptimizerLayers(),
        num_accumulation_batches: int = 1,
        post_forward_calculations: Optional[Tuple[PostForwardCalculation]] = None,
    ):
        # Initialize the base class
        super().__init__(
            name=name,
            dataloader=dataloader,
            extra_training_metrics=extra_training_metrics,
            device=device,
            loss=loss,
            post_forward_calculations=post_forward_calculations,
        )

        # Initialize subclass-specific attributes
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._set_training_layers_fn = set_training_layers_fn
        self._num_accumulation_batches = num_accumulation_batches

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        return self._scheduler

    @property
    def set_training_layers_fn(self) -> SetTrainingLayers:
        return self._set_training_layers_fn

    @property
    def num_accumulation_batches(self) -> int:
        return self._num_accumulation_batches

    @contextmanager
    def phase_settings(self, model: ProtoPNet):
        initial_states = {
            name: (param, param.requires_grad)
            for name, param in model.named_parameters()
        }
        try:
            assert (
                torch.is_grad_enabled()
            ), "Model should be in training mode with gradients enabled. Somewhere grad is being disabled outside of this phase."
            self.set_training_layers_fn.set_training_layers(model, self)
            yield
        finally:
            for _, (param, requires_grad) in initial_states.items():
                param.requires_grad = requires_grad
