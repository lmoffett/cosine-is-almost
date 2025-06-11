import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from swandb.slurm.slurm_runner import SlurmConfig, SlurmRunner
from swandb.weights_and_biases.sweep_runners import SlurmWandBAgentRunner


@pytest.fixture
def slurm_config():
    """Fixture for SLURM configuration"""
    return SlurmConfig(partition="gpu", gres="gpu:1")


@pytest.fixture
def mock_slurm_runner():
    """Fixture for mocked SlurmRunner"""
    mock_runner = MagicMock(spec=SlurmRunner)
    # Set up default return value for run_script
    mock_runner.run_script.return_value = ["12345"]
    return mock_runner


@pytest.fixture
def wandb_agent_runner(mock_slurm_runner, temp_dir):
    """Fixture for WandBAgentRunner with mocked SlurmRunner"""
    return SlurmWandBAgentRunner(
        slurm_runner=mock_slurm_runner,
        experiment_dir=temp_dir
    )


def test_wandb_agent_runner_init(slurm_config, temp_dir):
    """Test WandBAgentRunner initialization"""
    # Test with explicit SlurmRunner
    slurm_runner = SlurmRunner(config=slurm_config)
    runner = SlurmWandBAgentRunner(slurm_runner=slurm_runner, experiment_dir=temp_dir)
    
    assert runner.slurm_runner is slurm_runner
    assert runner.experiment_dir == temp_dir
    
    # Test with implicit SlurmRunner creation
    runner = SlurmWandBAgentRunner(slurm_config=slurm_config, experiment_dir=temp_dir)
    
    assert isinstance(runner.slurm_runner, SlurmRunner)
    assert runner.experiment_dir == temp_dir


def test_run_wandb_agents_basic(wandb_agent_runner, mock_slurm_runner):
    """Test running a single W&B agent"""
    # Run a W&B agent
    job_ids = wandb_agent_runner.run_wandb_agents("my-sweep-123")
    
    # Verify SlurmRunner.run_script was called with correct parameters
    mock_slurm_runner.run_script.assert_called_once()
    args, kwargs = mock_slurm_runner.run_script.call_args
    
    # Check command
    assert kwargs['script'] == "wandb agent my-sweep-123"
    
    # Check job ID
    assert kwargs['job_id'] == "wba-my-sweep-123"
    
    # Verify log directory was created in correct location
    assert "log_dir" in kwargs
    
    # Verify job IDs are returned correctly
    assert job_ids == ["12345"]


def test_run_wandb_agents_array(wandb_agent_runner, mock_slurm_runner):
    """Test running multiple W&B agents as array job"""
    # Run multiple W&B agents
    job_ids = wandb_agent_runner.run_wandb_agents("my-sweep-123", array_size=4)
    
    # Verify array_size was passed to SlurmRunner
    mock_slurm_runner.run_script.assert_called_once()
    args, kwargs = mock_slurm_runner.run_script.call_args
    
    assert kwargs["array_size"] == 4
    
    # Verify job IDs
    assert job_ids == ["12345"]


def test_run_wandb_agents_with_env(wandb_agent_runner, mock_slurm_runner):
    """Test running agent with custom environment variables"""
    # Custom environment
    custom_env = {"WANDB_PROJECT": "my-project", "CUDA_VISIBLE_DEVICES": "0", "WANDB_SERVICE": "something"}
    
    # Run with custom environment
    job_ids = wandb_agent_runner.run_wandb_agents("my-sweep-123", env=custom_env)
    
    # Verify environment was passed correctly
    mock_slurm_runner.run_script.assert_called_once()
    args, kwargs = mock_slurm_runner.run_script.call_args
    
    # Check that environment variables were passed
    assert "env" in kwargs
    assert kwargs["env"]["WANDB_PROJECT"] == "my-project"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    
    # Check that WANDB_SERVICE was removed
    assert "WANDB_SERVICE" not in kwargs["env"]
    
    # Check for W&B-specific environment variables
    assert "SWANDB_ARTIFACT_DIR" in kwargs["env"]
    assert "SWANDB_LOG_DIR" in kwargs["env"]


def test_wandb_directories_created(wandb_agent_runner, mock_slurm_runner, temp_dir):
    """Test that W&B directories are created correctly"""
    # Run agent
    sweep_id = "my-sweep-123"
    wandb_agent_runner.run_wandb_agents(sweep_id)
    
    # Verify directories were created
    sweep_dir = temp_dir / "sweeps" / sweep_id
    log_dir = sweep_dir / "logs"
    artifacts_dir = sweep_dir / "artifacts"
    
    # Note: In the real implementation, these would be created
    # Here we're just checking that the paths are correct in the mock call
    
    args, kwargs = mock_slurm_runner.run_script.call_args
    assert kwargs["script_dir"] == sweep_dir
    assert kwargs["log_dir"] == log_dir
    assert kwargs["env"]["SWANDB_ARTIFACT_DIR"] == str(artifacts_dir)
    assert kwargs["env"]["SWANDB_LOG_DIR"] == str(log_dir)


# Now let's do an integration test with real SlurmRunner (not mocked)

@patch("subprocess.run")
def test_integration_with_slurm_runner(mock_run, temp_dir, slurm_config):
    """Integration test with actual SlurmRunner (not mocked)"""
    # Mock subprocess.run
    mock_run.return_value = Mock(stdout="Submitted batch job 12345\n", returncode=0)
    
    # Create actual SlurmRunner
    slurm_runner = SlurmRunner(config=slurm_config)
    
    # Create WandBAgentRunner with real SlurmRunner
    runner = SlurmWandBAgentRunner(
        slurm_runner=slurm_runner,
        experiment_dir=temp_dir
    )
    
    # Run W&B agent
    sweep_id = "my-sweep-456"
    job_ids = runner.run_wandb_agents(sweep_id)
    
    # Verify subprocess.run was called
    mock_run.assert_called_once()
    
    # Verify sbatch script was created in the expected location
    sweep_dir = temp_dir / "sweeps" / sweep_id
    script_path = sweep_dir / f"wba-{sweep_id}.sbatch"
    assert script_path.exists()
    
    # Verify script content contains "wandb agent" command
    script_content = script_path.read_text()
    assert f"wandb agent {sweep_id}" in script_content
    
    # Verify job ID was returned
    assert job_ids == ["12345"]


@patch("subprocess.run")
def test_integration_error_handling(mock_run, temp_dir, slurm_config):
    """Test error handling in integration with SlurmRunner"""
    # Mock subprocess.run to raise an error
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd=["sbatch"],
        output="Error: Invalid partition name specified",
        stderr="Error: Invalid partition name specified"
    )
    
    # Create WandBAgentRunner with real SlurmRunner
    slurm_runner = SlurmRunner(config=slurm_config)
    runner = SlurmWandBAgentRunner(
        slurm_runner=slurm_runner,
        experiment_dir=temp_dir
    )
    
    # Verify error is propagated
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        runner.run_wandb_agents("my-sweep-789")
    
    assert "Invalid partition name" in str(exc_info.value.stderr)