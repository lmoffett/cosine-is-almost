import pytest
import torch
from unittest.mock import Mock, patch

from typing import Tuple, List

from protopnet.train.scheduling.early_stopping import CombinedEarlyStopping, SweepProjectEarlyStopping, ProjectPatienceEarlyStopping
from protopnet.train.scheduling.types import PostPhaseSummary, PrePhaseSummary


@pytest.fixture
def early_stopping_default():
    """Fixture for ProjectPatienceEarlyStopping with default parameters"""
    return ProjectPatienceEarlyStopping(project_patience=2)


@pytest.fixture
def early_stopping_custom():
    """Fixture for ProjectPatienceEarlyStopping with custom parameters"""
    return ProjectPatienceEarlyStopping(
        project_patience=3,
        post_project_threshold=0.75,
        project_phase_name="custom_project",
    )


@pytest.fixture
def pre_phase_summary():
    """Fixture for PrePhaseSummary with basic parameters"""
    return PrePhaseSummary(
        name="project",
        first_step=0,
        initial_target_metric=("accuracy", 0.5),
        expected_last_step=100,
    )


@pytest.fixture
def post_phase_summary(pre_phase_summary):
    """Fixture for PostPhaseSummary with improving metrics"""
    return pre_phase_summary.complete(
        last_step=100, final_target_metric=("accuracy", 0.8)
    )


def test_project_patience_early_stopping_initialization_with_defaults():
    """Test initialization of ProjectPatienceEarlyStopping with default parameters"""
    early_stopping = ProjectPatienceEarlyStopping(project_patience=2)
    assert early_stopping.project_patience == 2
    assert early_stopping.post_project_threshold == 0.0
    assert early_stopping.project_phase_name == "project"
    assert early_stopping.best_preproject_metric == float("-inf")
    assert early_stopping.best_project_metric == float("-inf")
    assert early_stopping.phases_without_improvement == 0


def test_project_patience_early_stopping_initialization_with_custom_parameters():
    """Test initialization of ProjectPatienceEarlyStopping with custom parameters"""
    early_stopping = ProjectPatienceEarlyStopping(
        project_patience=3,
        post_project_threshold=0.75,
        project_phase_name="custom_project",
    )
    assert early_stopping.project_patience == 3
    assert early_stopping.post_project_threshold == 0.75
    assert early_stopping.project_phase_name == "custom_project"


def test_step_end_should_stop_always_returns_false(
    early_stopping_default, pre_phase_summary
):
    """Test that step_end_should_stop always returns False regardless of input"""
    should_stop, reason = early_stopping_default.step_end_should_stop(
        step=50, current_step_target_metric=0.6, phase_summary=pre_phase_summary
    )
    assert not should_stop
    assert reason == "ProjectPatienceEarlyStopping does not stop after steps."


def test_phase_end_should_stop_non_project_phase(early_stopping_default):
    """Test phase_end_should_stop returns False for non-project phases"""

    non_project_summary = PostPhaseSummary(
        name="eval",
        first_step=0,
        initial_target_metric=("accuracy", 0.5),
        expected_last_step=100,
        last_step=100,
        final_target_metric=("accuracy", 0.8),
    )

    should_stop, reason = early_stopping_default.phase_end_should_stop(
        non_project_summary
    )
    assert not should_stop
    assert reason == "Not the project phase."


def test_phase_end_should_stop_with_improving_metrics(
    early_stopping_default, post_phase_summary
):
    """Test phase_end_should_stop with improving metrics doesn't trigger stopping"""
    should_stop, reason = early_stopping_default.phase_end_should_stop(
        post_phase_summary
    )
    assert not should_stop
    assert reason == "No stopping criteria met."
    assert early_stopping_default.phases_without_improvement == 0
    assert early_stopping_default.best_project_metric == 0.8
    assert early_stopping_default.best_preproject_metric == 0.5


def test_phase_end_should_stop_exceeds_patience(
    early_stopping_default, pre_phase_summary
):
    """Test phase_end_should_stop triggers stopping when project patience is exceeded"""
    # First phase
    summary_1 = pre_phase_summary.complete(
        last_step=100, final_target_metric=("accuracy", 0.6)
    )
    should_stop, _ = early_stopping_default.phase_end_should_stop(summary_1)
    assert not should_stop

    # Second phase with no improvement
    summary_2 = PrePhaseSummary(
        name="project",
        first_step=101,
        initial_target_metric=("accuracy", 0.4),
        expected_last_step=200,
    ).complete(last_step=200, final_target_metric=("accuracy", 0.5))
    should_stop, _ = early_stopping_default.phase_end_should_stop(summary_2)
    assert not should_stop

    # Third phase, but not project phase
    summary_3_joint = PrePhaseSummary(
        name="joint",
        first_step=101,
        initial_target_metric=("accuracy", 0.4),
        expected_last_step=200,
    ).complete(last_step=200, final_target_metric=("accuracy", 0.5))
    should_stop, _ = early_stopping_default.phase_end_should_stop(summary_3_joint)
    assert not should_stop

    # Third phase with no improvement - should trigger stopping
    summary_3_project = PrePhaseSummary(
        name="project",
        first_step=201,
        initial_target_metric=("accuracy", 0.4),
        expected_last_step=300,
    ).complete(last_step=300, final_target_metric=("accuracy", 0.5))
    should_stop, reason = early_stopping_default.phase_end_should_stop(
        summary_3_project
    )
    assert should_stop
    assert reason == "Project phase patience exceeded."


def test_phase_end_should_stop_resets_patience_counter(
    early_stopping_default, pre_phase_summary
):
    """Test that phases_without_improvement counter resets when there's improvement"""
    # First phase with no improvement
    summary_1 = pre_phase_summary.complete(
        last_step=100, final_target_metric=("accuracy", 0.6)
    )
    early_stopping_default.phase_end_should_stop(summary_1)

    # Second phase with no improvement
    summary_2 = PrePhaseSummary(
        name="project",
        first_step=101,
        initial_target_metric=("accuracy", 0.4),
        expected_last_step=200,
    ).complete(last_step=200, final_target_metric=("accuracy", 0.5))
    early_stopping_default.phase_end_should_stop(summary_2)
    assert early_stopping_default.phases_without_improvement == 1

    # Third phase with improvement
    summary_3 = PrePhaseSummary(
        name="project",
        first_step=201,
        initial_target_metric=("accuracy", 0.6),
        expected_last_step=300,
    ).complete(last_step=300, final_target_metric=("accuracy", 0.7))
    should_stop, _ = early_stopping_default.phase_end_should_stop(summary_3)
    assert not should_stop
    assert early_stopping_default.phases_without_improvement == 0


def test_phase_end_should_stop_handles_tensor_metrics(
    early_stopping_default, pre_phase_summary
):
    """Test that phase_end_should_stop properly handles torch.Tensor metrics"""
    tensor_summary = pre_phase_summary.complete(
        last_step=100, final_target_metric=("accuracy", torch.tensor(0.8))
    )
    should_stop, reason = early_stopping_default.phase_end_should_stop(tensor_summary)
    assert not should_stop
    assert reason == "No stopping criteria met."
    assert early_stopping_default.best_project_metric == 0.8
    assert early_stopping_default.best_preproject_metric == 0.5


class MockEarlyStopping:
    """Mock early stopping class for testing."""
    def __init__(self, should_stop=False, reason="mock reason"):
        self.should_stop = should_stop
        self.reason = reason
        self.call_count = 0
    
    def step_end_should_stop(
        self,
        step: int,
        current_step_target_metric: float,
        phase_summary: PrePhaseSummary,
    ) -> Tuple[bool, str]:
        return self.should_stop, self.reason
    
    def phase_end_should_stop(
        self, phase_summary: PostPhaseSummary
    ) -> Tuple[bool, str]:
        self.call_count += 1
        return self.should_stop, self.reason
    
    def set_should_stop(self, should_stop: bool):
        """Helper to change the stopping behavior."""
        self.should_stop = should_stop

def test_combined_early_stopping_empty_dict():
    """Test that CombinedEarlyStopping raises error with empty dict."""
    with pytest.raises(ValueError, match="At least one early stopping criterion must be provided"):
        CombinedEarlyStopping({})

def test_combined_early_stopping_all_false():
    """Test when all stoppers return False."""
    stopper1 = MockEarlyStopping(should_stop=False, reason="not stopping 1")
    stopper2 = MockEarlyStopping(should_stop=False, reason="not stopping 2")
    
    combined = CombinedEarlyStopping({
        "stopper1": stopper1,
        "stopper2": stopper2
    })
    
    phase_summary = PostPhaseSummary(
        name="project",
        first_step=0,
        expected_last_step=100,
        initial_target_metric=(0, 0.0),
        last_step=100,
        final_target_metric=(1, 1.0)
    )
    
    should_stop, reason = combined.phase_end_should_stop(phase_summary)
    assert not should_stop
    assert "stopper1: not stopping 1" in reason
    assert "stopper2: not stopping 2" in reason
    assert stopper1.call_count == 1
    assert stopper2.call_count == 1

def test_combined_early_stopping_all_true():
    """Test when all stoppers return True."""
    stopper1 = MockEarlyStopping(should_stop=True, reason="stopping 1")
    stopper2 = MockEarlyStopping(should_stop=True, reason="stopping 2")
    
    combined = CombinedEarlyStopping({
        "stopper1": stopper1,
        "stopper2": stopper2
    })
    
    phase_summary = PostPhaseSummary(
        name="project",
        first_step=0,
        expected_last_step=100,
        initial_target_metric=(0, 0.0),
        last_step=100,
        final_target_metric=(1, 1.0)
    )
    
    should_stop, reason = combined.phase_end_should_stop(phase_summary)
    assert should_stop
    # Since we stop after the first True, we should only see stopper1's reason
    assert reason == "stopper1: stopping 1"
    assert stopper1.call_count == 1
    assert stopper2.call_count == 0

def test_combined_early_stopping_stateful():
    """Test that stoppers maintain state and can change behavior."""
    stopper1 = MockEarlyStopping(should_stop=False, reason="not stopping 1")
    stopper2 = MockEarlyStopping(should_stop=False, reason="not stopping 2")
    
    combined = CombinedEarlyStopping({
        "stopper1": stopper1,
        "stopper2": stopper2
    })
    
    phase_summary = PostPhaseSummary(
        name="project",
        first_step=0,
        expected_last_step=100,
        initial_target_metric=(0, 0.0),
        last_step=100,
        final_target_metric=(1, 1.0)
    )
    
    # First call - both return False
    should_stop, reason = combined.phase_end_should_stop(phase_summary)
    assert not should_stop
    assert stopper1.call_count == 1
    assert stopper2.call_count == 1
    
    # Change stopper1 to return True
    stopper1.set_should_stop(True)
    
    # Second call - stopper1 returns True
    should_stop, reason = combined.phase_end_should_stop(phase_summary)
    assert should_stop
    assert stopper1.call_count == 2
    # stopper2 shouldn't be called since stopper1 returned True
    assert stopper2.call_count == 1


# Mock classes to simulate W&B API responses
class MockWandbRun:
    def __init__(self, history_data: List[dict]):
        self.history_data = history_data
        
    def scan_history(self, keys=None):
        return self.history_data

class MockWandbSweep:
    def __init__(self, runs):
        self.runs = runs

@pytest.fixture
def mock_wandb_run():
    """Create a mock wandb.run with basic attributes"""
    run = Mock()
    run.entity = "test_entity"
    run.project = "test_project"
    run.sweep_id = "test_sweep"
    return run

# Input Validation Tests
def test_invalid_percentile_threshold():
    """Test that percentile_threshold must be between 0 and 100"""
    with pytest.raises(ValueError, match="Percentile threshold must be between 0 and 100"):
        SweepProjectEarlyStopping(percentile_threshold=-1)
    with pytest.raises(ValueError, match="Percentile threshold must be between 0 and 100"):
        SweepProjectEarlyStopping(percentile_threshold=101)

def test_invalid_min_runs():
    """Test that min_runs_required must be positive"""
    with pytest.raises(ValueError, match="Minimum required runs must be positive"):
        SweepProjectEarlyStopping(min_runs_required=0)

def test_initialization_defaults():
    """Test initialization with default values"""
    stopper = SweepProjectEarlyStopping()
    assert stopper.percentile_threshold == 30.0
    assert stopper.project_phase_name == "project"
    assert stopper.metric_name == "target_metric"
    assert stopper.min_runs_required == 10
    assert stopper.current_project_count == 0

# Non-Project Phase Tests
def test_non_project_phase():
    """Test behavior during non-project phases"""
    stopper = SweepProjectEarlyStopping()

    phase_summary = PostPhaseSummary(
        name="train", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=(0, 0.0),
        last_step=100,
        final_target_metric=(1, 1.0)
    )
    
    should_stop, reason = stopper.phase_end_should_stop(phase_summary)
    assert not should_stop
    assert "Not the project phase" in reason
    assert stopper.current_project_count == 0  # Shouldn't increment for non-project phases

# Project Phase Tests
@pytest.mark.parametrize("num_runs,should_stop", [
    (3, False),  # Below min_runs_required
    (5, True),   # Equal to min_runs_required, poor performance
    (7, True),   # Above min_runs_required, poor performance
])
@patch('wandb.Api')
@patch('wandb.run')
def test_project_phase_run_counts(mock_run, mock_api, num_runs, should_stop, mock_wandb_run):
    """Test behavior with different numbers of runs"""
    # Setup mock data
    mock_run.return_value = mock_wandb_run
    
    # Create mock run histories where each run has multiple project evaluations
    histories = []
    for i in range(num_runs):
        # Each run has metrics from multiple projects
        history = [
            # Project 0
            {'project/target_metric': 70.0 + i},
            # Project 1 (current project we're testing)
            {'project/target_metric': 80.0 + i},  # Current run will be worst
            # Project 2
            {'project/target_metric': 85.0 + i},
        ]
        histories.append(MockWandbRun(history))
    
    mock_sweep = MockWandbSweep(histories)
    mock_api.return_value.sweep.return_value = mock_sweep
    
    stopper = SweepProjectEarlyStopping(
        percentile_threshold=25.0,
        min_runs_required=5,
        brackets=[0]
    )

    phase_summary = PostPhaseSummary(
        name="project", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("accuracy", 0.0),
        last_step=100,
        final_target_metric=("accuracy", 60.0)  # Worst performance
    )
    
    should_stop_result, reason = stopper.phase_end_should_stop(phase_summary)
    
    if num_runs < 5:
        assert not should_stop_result
        assert f"Not enough runs for comparison (have {num_runs}" in reason
    else:
        assert should_stop_result
        assert f"Performance in bottom 25.0%" in reason

# Performance Comparison Tests
@pytest.mark.parametrize("current_metric,other_metrics,should_stop", [
    (92.0, [80.0, 85.0, 90.0, 95.0, 100.0], False),  # Above threshold
    (75.0, [80.0, 85.0, 90.0, 95.0, 100.0], True),  # Below threshold
    (0.0, [80.0, 85.0, 90.0, 95.0], False),  # Below threshold but not enough runs
    (85.0, [80.0, 85.0, 90.0, 95.0, 100.0], False),  # At 50th percentile
])
@patch('wandb.Api')
@patch('wandb.run')
def test_performance_comparisons(mock_run, mock_api, current_metric, other_metrics, 
                            should_stop, mock_wandb_run):
    """Test different performance scenarios"""
    mock_run.return_value = mock_wandb_run
    
    # Create mock histories
    histories = []
    for metric in other_metrics:
        history = [
            {'project/target_metric': metric},
            {'project/target_metric': metric + 1},
            {'project/target_metric': metric + 2},
        ]
        histories.append(MockWandbRun(history))
    
    mock_sweep = MockWandbSweep(histories)
    mock_api.return_value.sweep.return_value = mock_sweep
    
    stopper = SweepProjectEarlyStopping(percentile_threshold=25.0, min_runs_required=5, brackets=[0])
    
    phase_summary = PostPhaseSummary(
        name="project",
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("accuracy", 0.0),
        last_step=100,
        final_target_metric=("accuracy", current_metric)
    )

    should_stop_result, reason = stopper.phase_end_should_stop(phase_summary)
    assert should_stop_result == should_stop

@patch('wandb.Api')
@patch('wandb.run')
def test_no_sweep_id(mock_run, mock_api):
    """Test behavior when sweep_id is missing"""
    mock_run.entity = "test_entity"
    mock_run.project = "test_project"
    mock_run.sweep_id = None
    
    stopper = SweepProjectEarlyStopping(
        brackets=[0]
    )
    phase_summary = PostPhaseSummary(
        name="project", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("accuracy", 0.0),
        last_step=100,
        final_target_metric=("accuracy", 80.0)
    )
    
    with pytest.raises(ValueError, match="No active sweep found in current run"):
        stopper.phase_end_should_stop(phase_summary)

# Project Counter Tests
@patch('wandb.Api')
@patch('wandb.run')
def test_project_counter_increment(mock_run, mock_api, mock_wandb_run):
    """Test that project counter increments correctly"""
    mock_run.return_value = mock_wandb_run
    
    histories = [MockWandbRun([{'project/target_metric': 90.0}])]
    mock_sweep = MockWandbSweep(histories)
    mock_api.return_value.sweep.return_value = mock_sweep
    
    stopper = SweepProjectEarlyStopping()
    phase_summary = PostPhaseSummary(
        name="project", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("accuracy", 0.0),
        last_step=100,
        final_target_metric=("accuracy", 80.0)
    )
    
    assert stopper.current_project_count == 0
    stopper.phase_end_should_stop(phase_summary)
    assert stopper.current_project_count == 1
    stopper.phase_end_should_stop(phase_summary)
    assert stopper.current_project_count == 2

# Project Counter Tests
@patch('wandb.Api')
@patch('wandb.run')
def test_bracketing(mock_run, mock_api, mock_wandb_run):
    """Test that project counter increments correctly"""
    mock_run.return_value = mock_wandb_run
    
    histories = []
    other_run_metric = 90.0
    for _ in range(3):
        history = [
            {'project/target_metric': other_run_metric},
            {'project/target_metric': other_run_metric + 1},
            {'project/target_metric': other_run_metric + 2},
        ]
        histories.append(MockWandbRun(history))
    
    mock_sweep = MockWandbSweep(histories)
    mock_api.return_value.sweep.return_value = mock_sweep
    
    stopper = SweepProjectEarlyStopping(
        brackets=[1],
        min_runs_required=1
    )
    phase_summary = PostPhaseSummary(
        name="project", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("target_metric", 0.0),
        last_step=100,
        final_target_metric=("target_metric", 80.0)
    )
    
    assert stopper.current_project_count == 0
    should_stop_0, should_stop_0_message = stopper.phase_end_should_stop(phase_summary)
    assert should_stop_0 is False
    assert should_stop_0_message == "Skipping percentile check at project at non-bracket project 0"
    
    assert stopper.current_project_count == 1
    should_stop_1, should_stop_1_message = stopper.phase_end_should_stop(phase_summary)
    assert should_stop_1 is True
    assert should_stop_1_message == "Performance in bottom 30.0% (compared against 3 runs at project 1)"


# Metric Name Tests
@patch('wandb.Api')
@patch('wandb.run')
def test_custom_metric_name(mock_run, mock_api, mock_wandb_run):
    """Test handling of custom metric names"""
    mock_run.return_value = mock_wandb_run
    
    # Mock history with custom metric name
    histories = [MockWandbRun([{'project/custom_metric': 90.0}])]
    mock_sweep = MockWandbSweep(histories)
    mock_api.return_value.sweep.return_value = mock_sweep
    
    stopper = SweepProjectEarlyStopping(metric_name="custom_metric")
    phase_summary = PostPhaseSummary(
        name="project", # Not a project phase
        first_step=0,
        expected_last_step=100,
        initial_target_metric=("accuracy", 0.0),
        last_step=100,
        final_target_metric=("accuracy", 80.0)
    )
    
    should_stop, reason = stopper.phase_end_should_stop(phase_summary)
    assert not should_stop  # Should continue as it's performing well