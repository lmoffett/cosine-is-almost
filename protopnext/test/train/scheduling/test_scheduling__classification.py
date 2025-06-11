import pytest
import torch
import torch.utils.data

from protopnet.models.prototree import ProtoTreeBackpropPhaseWithLeafUpdate
from protopnet.models.st_protopnet import (
    STProtoPNetBackpropPhase,
    STProtoPNetClassificationPhase,
)
from protopnet.train.scheduling.scheduling import (
    ClassificationBackpropPhase,
    ClassificationInferencePhase,
)
from protopnet.train.scheduling.types import StepContext


@pytest.fixture
def mock_loss():
    class MockBatchLoss:
        def required_forward_results(self):
            return ["prototype_activations", "logits"]

    class MockLoss:
        def __init__(self, requires_grad=True):
            self.batch_loss = MockBatchLoss()
            self.requires_grad = requires_grad

        def __call__(self, target, model, **kwargs):
            # Return a dummy loss value
            return (
                torch.tensor(0.71 * len(target), requires_grad=self.requires_grad),
                {},
            )

    return MockLoss()


@pytest.fixture
def targets():
    return torch.tensor([0, 1, 2, 0, 1])


@pytest.fixture
def imgs():
    return torch.randn(5, 3, 32, 32)


@pytest.fixture
def mock_dataset(imgs, targets):
    assert len(imgs) == len(targets)

    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(imgs)

        def __getitem__(self, idx):
            return {"img": imgs[idx], "target": targets[idx]}

    return MockDataset()


@pytest.fixture
def mock_dataloader(mock_dataset):
    return torch.utils.data.DataLoader(mock_dataset, batch_size=2, shuffle=False)


@pytest.fixture
def mock_post_forward_calculation():
    class MockPostForwardCalc:
        def __call__(self, model, **kwargs):
            return {"mock_calc_result": torch.tensor(1.0)}

    return MockPostForwardCalc()


@pytest.fixture(params=["vanilla", "prototree", "stprotopnet"])
def mock_model(request, mock_protopnet, mock_prototree, mock_stprotopnet):
    """Returns just the model instance based on the parameterized type."""
    if request.param == "vanilla":
        return mock_protopnet
    elif request.param == "prototree":
        return mock_prototree
    elif request.param == "stprotopnet":
        return mock_stprotopnet


@pytest.fixture()
def classification_no_backprop_phase(
    mock_model,
    mock_stprotopnet,
    mock_dataloader,
    mock_loss,
    mock_post_forward_calculation,
):
    if mock_model == mock_stprotopnet:
        return STProtoPNetClassificationPhase(
            name="test_phase",
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
            support_loss=mock_loss,
            trivial_loss=mock_loss,
            post_forward_calculations=(mock_post_forward_calculation,),
        )
    else:
        return ClassificationInferencePhase(
            name="test_phase",
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
            loss=mock_loss,
            post_forward_calculations=(mock_post_forward_calculation,),
        )


@pytest.fixture()
def classification_backprop_phase(
    mock_protopnet, mock_loss, mock_dataloader, mock_post_forward_calculation
):
    optimizer = torch.optim.SGD(mock_protopnet.parameters(), lr=0.01)
    return ClassificationBackpropPhase(
        name="test_phase",
        dataloader=mock_dataloader,
        device=torch.device("cpu"),
        loss=mock_loss,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        post_forward_calculations=(mock_post_forward_calculation,),
        num_accumulation_batches=2,
    )


@pytest.fixture()
def prototree_backprop_phase(
    mock_prototree, mock_loss, mock_dataloader, mock_post_forward_calculation
):
    optimizer = torch.optim.SGD(mock_prototree.parameters(), lr=0.01)
    return ProtoTreeBackpropPhaseWithLeafUpdate(
        name="test_phase",
        dataloader=mock_dataloader,
        device=torch.device("cpu"),
        loss=mock_loss,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        post_forward_calculations=(mock_post_forward_calculation,),
        num_accumulation_batches=2,
    )


@pytest.fixture()
def st_protopnet_backprop_phase(
    mock_stprotopnet, mock_loss, mock_dataloader, mock_post_forward_calculation
):
    optimizer = torch.optim.SGD(mock_stprotopnet.parameters(), lr=0.01)
    return STProtoPNetBackpropPhase(
        name="test_phase",
        dataloader=mock_dataloader,
        device=torch.device("cpu"),
        support_loss=mock_loss,
        trivial_loss=mock_loss,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
        post_forward_calculations=(mock_post_forward_calculation,),
        num_accumulation_batches=2,
    )


@pytest.fixture()
def backprop_phase(
    mock_model,
    mock_protopnet,
    mock_prototree,
    mock_stprotopnet,
    classification_backprop_phase,
    prototree_backprop_phase,
    st_protopnet_backprop_phase,
):
    """Parameterized fixture that provides both classification and prototree backprop phases for testing."""
    if mock_model == mock_protopnet:
        return classification_backprop_phase
    elif mock_model == mock_prototree:
        return prototree_backprop_phase
    elif mock_model == mock_stprotopnet:
        return st_protopnet_backprop_phase


@pytest.fixture(params=["inference", "classification"])
def classification_phase(
    request,
    classification_no_backprop_phase,
    backprop_phase,
):
    """Parameterized fixture that provides inference and both types of backprop phases for testing."""
    if request.param == "inference":
        return classification_no_backprop_phase
    elif request.param == "classification":
        return backprop_phase


def test_classification_phase_initialization(classification_phase):
    """Test that the inference phase is initialized correctly."""
    assert classification_phase.name == "test_phase"
    assert isinstance(classification_phase.dataloader, torch.utils.data.DataLoader)
    assert classification_phase.device == torch.device("cpu")
    assert len(classification_phase.post_forward_calculations) == 1
    assert classification_phase.forward_calc_flags["return_prototype_layer_output_dict"]


def test_append_extra_forward_results(classification_phase, mock_model, targets):
    """Test that extra forward results are appended correctly."""
    input_dict = {"img": torch.randn(2, 3, 4, 5)}
    output_dict = {
        "logits": torch.randn(2, 3),
        "similarity_score_to_each_prototype": torch.randn(2, 4, 4, 5),
        "target": targets[:2],
    }

    result = classification_phase.append_extra_forward_results(
        model=mock_model, input=input_dict, output=output_dict, phase="train"
    )

    assert "mock_calc_result" in result
    for key in output_dict:
        assert key in result
        assert result[key] is output_dict[key]
    assert isinstance(result["mock_calc_result"], torch.Tensor)
    assert result != output_dict  # Should be a new dictionary


def test_run_classification(
    classification_phase, mock_model, mock_stprotopnet, imgs, targets
):
    """Test the run_classification method."""
    batch_data = {"img": imgs[:2], "target": targets[:2]}
    metrics_dict = {}

    output, loss = classification_phase.run_classification(
        model=mock_model,
        batch_data_dict=batch_data,
        epoch_metrics_dict=metrics_dict,
    )

    if mock_model == mock_stprotopnet:
        output = output[0]

    assert isinstance(output, dict)
    assert "similarity_score_to_each_prototype" in output
    assert "logits" in output

    assert isinstance(loss, torch.Tensor)
    assert "n_examples" in metrics_dict
    assert "n_correct" in metrics_dict
    assert metrics_dict["n_examples"] == 2
    assert "accu" in metrics_dict
    assert (
        metrics_dict["accu"] == metrics_dict["n_correct"] / metrics_dict["n_examples"]
    )
    assert loss == 1.42


def test_run_step_happy_path(classification_phase, mock_model):
    """Test the run_step method."""
    step_context = StepContext(global_step=0, step_in_phase=0)

    metrics = classification_phase.run_step(model=mock_model, step_context=step_context)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert metrics["accuracy"] <= 1.0 and metrics["accuracy"] >= 0.0
    assert "total_loss" in metrics
    assert metrics["total_loss"] == pytest.approx(1.183333)
    assert metrics["accuracy"] == metrics["accu"]


def test_phase_settings_context(classification_phase, mock_model):
    """Test that the phase_settings context manager works correctly."""
    original_state = mock_model.training

    should_grad = isinstance(classification_phase, ClassificationBackpropPhase)

    with classification_phase.phase_settings(mock_model):
        assert mock_model.training == should_grad
        assert torch.is_grad_enabled() == should_grad

    assert mock_model.training == original_state


def test_error_handling_nan_loss(classification_phase, mock_model, mock_stprotopnet):
    """Test that NaN losses are handled appropriately."""

    class NanLoss:
        def __init__(self):
            self.batch_loss = self

        def required_forward_results(self):
            return []

        def __call__(self, *args, **kwargs):
            return (torch.tensor(float("nan")), {})

    if mock_model == mock_stprotopnet:
        phase = STProtoPNetClassificationPhase(
            name="test",
            dataloader=classification_phase.dataloader,
            device=torch.device("cpu"),
            support_loss=NanLoss(),
            trivial_loss=NanLoss(),
        )
    else:
        phase = ClassificationInferencePhase(
            name="test",
            dataloader=classification_phase.dataloader,
            device=torch.device("cpu"),
            loss=NanLoss(),
        )

    with pytest.raises(ValueError, match="NaN values"):
        phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))


def test_inference_training_state_for_step(
    mock_dataloader, mock_loss, mock_model, mock_stprotopnet
):
    """Test that inference phase properly manages training state without optimization."""

    if mock_model == mock_stprotopnet:
        phase = STProtoPNetClassificationPhase(
            name="test",
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
            support_loss=mock_loss,
            trivial_loss=mock_loss,
        )
    else:
        phase = ClassificationInferencePhase(
            name="test",
            dataloader=mock_dataloader,
            device=torch.device("cpu"),
            loss=mock_loss,
        )

    # Track training states throughout run_step
    training_states = []
    original_run_classification = phase.run_classification

    # Set model to training mode initially
    mock_model.train()
    initial_state = mock_model.training

    def mock_run_classification(*args, **kwargs):
        training_states.append(mock_model.training)
        return original_run_classification(*args, **kwargs)

    phase.run_classification = mock_run_classification

    # Run step
    with phase.phase_settings(mock_model):
        phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))

    # Verify model was in eval mode during run_step
    assert all(not state for state in training_states)
    # Verify model returns to original state
    assert mock_model.training == initial_state


@pytest.fixture
def mock_step_tracking():
    """Fixture to track optimizer and gradient operations."""

    class StepTracker:
        def __init__(self):
            self.optimizer_steps = []
            self.grad_zeros = []
            self.backward_calls = []

        def mock_step(self, original_step, backward_calls):
            def _step():
                self.optimizer_steps.append(len(backward_calls))
                return original_step()

            return _step

        def mock_zero_grad(self, original_zero_grad, backward_calls):
            def _zero_grad():
                self.grad_zeros.append(len(backward_calls))
                return original_zero_grad()

            return _zero_grad

    return StepTracker()


def test_device_movement(mock_model, backprop_phase):
    """Test that tensors are correctly moved to the specified device."""
    phase = backprop_phase

    moved_tensors = []
    original_run_classification = phase.run_classification

    def mock_run_classification(model, batch_data_dict, epoch_metrics_dict):
        # Track devices of all tensors in the batch
        moved_tensors.append(
            {
                key: tensor.device
                for key, tensor in batch_data_dict.items()
                if isinstance(tensor, torch.Tensor)
            }
        )
        return original_run_classification(model, batch_data_dict, epoch_metrics_dict)

    phase.run_classification = mock_run_classification

    # Run a training step
    phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))

    # Verify all tensors were moved to the correct device
    for batch_devices in moved_tensors:
        assert all(
            device == phase.device for device in batch_devices.values()
        ), "Not all tensors were moved to the correct device"


def test_gradient_state_management(mock_model, backprop_phase):
    """Test gradients are set back to original state after training step."""
    phase = backprop_phase

    gradient_states = []
    original_set_training = phase.set_training_layers_fn.set_training_layers

    set_training_calls = []

    def mock_set_training(model, current_phase):
        # Capture gradient states before and after setting training layers
        set_training_calls.append(current_phase)
        gradient_states.append(
            {name: param.requires_grad for name, param in model.named_parameters()}
        )
        result = original_set_training(model, current_phase)
        gradient_states.append(
            {name: param.requires_grad for name, param in model.named_parameters()}
        )
        return result

    phase.set_training_layers_fn.set_training_layers = mock_set_training

    # Store initial gradient states
    initial_states = {
        name: param.requires_grad for name, param in mock_model.named_parameters()
    }

    with phase.phase_settings(mock_model):
        # Run training step
        phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))

    # Check final states match initial
    final_states = {
        name: param.requires_grad for name, param in mock_model.named_parameters()
    }

    assert len(set_training_calls) > 0, "set_training_layers_fn was not called"
    assert (
        set_training_calls[0] == phase
    ), "set_training_layers_fn was called with wrong phase"
    assert gradient_states[0] == initial_states, "Initial gradient states were modified"
    assert final_states == initial_states, "Gradient states weren't properly restored"
    assert len(gradient_states) >= 2, "Training layers weren't properly configured"


def test_backprop_optimization_timing(mock_model, mock_dataloader, backprop_phase):
    """Test that optimizer steps, grad clearing, and backwards happen at the correct times during training."""
    phase = backprop_phase

    # Track all relevant calls
    backward_calls = []
    optimizer_steps = []
    grad_zeros = []

    # Store original method to track calls
    original_handle = phase._handle_loss_and_optimization

    def mock_handle(loss, batch_idx):
        if hasattr(phase, "optimizer"):
            loss.backward(retain_graph=True)
            backward_calls.append(True)  # Record backward call

        if ((batch_idx + 1) % phase.num_accumulation_batches == 0) or (
            batch_idx + 1 == len(mock_dataloader)
        ):
            if hasattr(phase, "optimizer"):
                optimizer_steps.append(batch_idx)
                phase.optimizer.step()
                grad_zeros.append(batch_idx)
                phase.optimizer.zero_grad()

    phase._handle_loss_and_optimization = mock_handle

    try:
        phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))
    finally:
        # Restore original method
        phase._handle_loss_and_optimization = original_handle

    # Verify optimization timing
    assert len(backward_calls) == 3, "Backward should be called after every batch"
    assert optimizer_steps == [1, 2], "Optimizer should step after batches 1, 2"
    assert (
        grad_zeros == optimizer_steps
    ), "Gradient clearing should match optimizer steps"


@pytest.mark.xfail(reason="Fix for prototree")
def test_optimizer_callbacks(mock_model, backprop_phase, mock_dataloader):
    """Test pre/post optimizer callbacks are called appropriately."""
    phase = backprop_phase

    callback_tracker = {"pre": [], "post": []}

    def track_pre(dataloader, optimizer):
        callback_tracker["pre"].append((dataloader, optimizer))

    def track_post(output, target, dataloader):
        callback_tracker["post"].append((output, target, dataloader))

    phase._pre_optimizer_model_update = track_pre
    phase._post_optimizer_model_update = track_post

    phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))

    assert (
        len(callback_tracker["pre"]) == 1
    ), "Pre-optimizer callback should be called once"
    assert (
        callback_tracker["pre"][0][0] == mock_dataloader
    ), "Wrong dataloader in pre callback"
    assert (
        callback_tracker["pre"][0][1] == phase.optimizer
    ), "Wrong optimizer in pre callback"

    expected_post_calls = len(mock_dataloader) // phase.num_accumulation_batches
    assert expected_post_calls == 1
    assert (
        len(callback_tracker["post"]) == expected_post_calls
    ), f"Expected {expected_post_calls} post-optimizer callbacks"

    for post_call in callback_tracker["post"]:
        assert post_call[2] == mock_dataloader, "Wrong dataloader in post callback"


def test_scheduler_step(mock_model, backprop_phase):
    """Test scheduler steps exactly once per training step."""
    phase = backprop_phase

    scheduler_steps = 0
    original_step = phase.scheduler.step

    def count_steps():
        nonlocal scheduler_steps
        scheduler_steps += 1
        return original_step()

    phase.scheduler.step = count_steps

    phase.run_step(mock_model, StepContext(global_step=0, step_in_phase=0))
    assert scheduler_steps == 1, "Scheduler should step exactly once per training step"
