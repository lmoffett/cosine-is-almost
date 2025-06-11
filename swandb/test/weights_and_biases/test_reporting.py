import pandas as pd
import pytest
from click.testing import CliRunner
from unittest import mock
from pathlib import Path

from swandb.weights_and_biases.reporting import reporting

@pytest.fixture
def runner():
    """Fixture for invoking command-line interfaces."""
    return CliRunner()

@pytest.fixture
def mock_wandb_api():
    """Mock the wandb.Api() class."""
    with mock.patch('wandb.Api') as mock_api:
        yield mock_api

@pytest.fixture
def mock_runs_response():
    """Create mock runs for testing export_runs command."""
    # Create mock run objects
    runs = []
    for i in range(3):
        mock_run = mock.MagicMock()
        mock_run.name = f"test-run-{i}"
        
        # Mock summary with some test metrics
        mock_summary = mock.MagicMock()
        mock_summary._json_dict = {"accuracy": 0.9 + i*0.01, "loss": 0.1 - i*0.01}
        mock_run.summary = mock_summary
        
        # Mock config with hyperparameters
        mock_run.config = {
            "learning_rate": 0.001 * (i+1),
            "batch_size": 32 * (i+1),
            "_internal_param": "should_be_filtered"
        }
        
        # Set sweep property
        if i == 0:  # Only first run has a sweep
            mock_sweep = mock.MagicMock()
            mock_sweep.id = "sweep123"
            mock_run.sweep = mock_sweep
        else:
            mock_run.sweep = None
            
        runs.append(mock_run)
    
    return runs

@pytest.fixture
def mock_sweeps_response():
    """Create mock sweeps for testing export_sweeps command."""
    sweeps = []
    for i in range(2):
        mock_sweep = mock.MagicMock()
        mock_sweep.id = f"sweep-{i}"
        mock_sweep.name = f"test-sweep-{i}"
        
        # Mock sweep config
        mock_sweep.config = {
            "method": "random" if i == 0 else "bayes",
            "metric": {"name": "accuracy", "goal": "maximize"},
            "_internal_key": "should_be_filtered"
        }
        
        # Mock best run method
        best_run = mock.MagicMock()
        best_run.id = f"best-run-{i}"
        best_run.name = f"best-run-name-{i}"
        mock_sweep.best_run.return_value = best_run
        
        sweeps.append(mock_sweep)
    
    return sweeps


def test_export_runs_default_params(runner, mock_wandb_api, mock_runs_response, temp_dir):
    """Test export_runs command with default parameters."""
    # Set up the mock API
    api_instance = mock_wandb_api.return_value
    api_instance.runs.return_value = mock_runs_response
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command with required parameters (entity and project)
        result = runner.invoke(
            reporting, 
            ["runs", "--wandb-entity", "test-entity", "--wandb-project", "test-project"]
        )
        
        # Verify the command executed successfully
        assert result.exit_code == 0
        
        # Check API was called correctly
        api_instance.runs.assert_called_once_with("test-entity/test-project")
        
        # Verify output file was created with expected data
        output_file = Path("wandb-runs.csv")
        assert output_file.exists()
        
        # Load and check the CSV content
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert "accuracy" in df.columns
        assert "loss" in df.columns
        assert "learning_rate" in df.columns
        assert "batch_size" in df.columns
        assert "sweep_id" in df.columns
        assert "_internal_param" not in df.columns  # Should be filtered out

def test_export_runs_custom_params(runner, mock_wandb_api, mock_runs_response, temp_dir):
    """Test export_runs command with custom parameters."""
    # Set up the mock API
    api_instance = mock_wandb_api.return_value
    api_instance.runs.return_value = mock_runs_response
    
    custom_output = "custom-runs.csv"
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command with custom parameters
        result = runner.invoke(
            reporting, 
            [
                "runs", 
                "--wandb-entity", "custom-entity", 
                "--wandb-project", "custom-project",
                "--output", custom_output,
                "--internal-metadata"  # Include internal metadata
            ]
        )
        
        # Verify the command executed successfully
        assert result.exit_code == 0
        
        # Check API was called correctly
        api_instance.runs.assert_called_once_with("custom-entity/custom-project")
        
        # Verify custom output file was created
        output_file = Path(custom_output)
        assert output_file.exists()
        
        # Load and check the CSV content
        df = pd.read_csv(output_file)
        assert len(df) == 3
        assert "_internal_param" in df.columns  # Should be included now

def test_export_runs_no_runs(runner, mock_wandb_api, temp_dir):
    """Test export_runs command when no runs are found."""
    # Set up the mock API to return empty list
    api_instance = mock_wandb_api.return_value
    api_instance.runs.return_value = []
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command
        result = runner.invoke(
            reporting, 
            ["runs", "--wandb-entity", "test-entity", "--wandb-project", "test-project"]
        )
        
        # Verify the command executed successfully but with warning
        assert result.exit_code == 0
        
        # Verify output file was not created
        output_file = Path("wandb-runs.csv")
        assert not output_file.exists()


def test_export_sweeps_default_params(runner, mock_wandb_api, mock_sweeps_response, temp_dir):
    """Test export_sweeps command with default parameters."""
    # Set up the mock API
    api_instance = mock_wandb_api.return_value
    mock_project = mock.MagicMock()
    mock_project.sweeps.return_value = mock_sweeps_response
    api_instance.project.return_value = mock_project
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command
        result = runner.invoke(
            reporting, 
            ["sweeps", "--wandb-entity", "test-entity", "--wandb-project", "test-project"]
        )
        
        # Verify the command executed successfully
        assert result.exit_code == 0
        
        # Check API was called correctly
        api_instance.project.assert_called_once_with(name="test-project", entity="test-entity")
        mock_project.sweeps.assert_called_once()
        
        # Verify output file was created
        output_file = Path("wandb-sweeps.csv")
        assert output_file.exists()
        
        # Load and check the CSV content
        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert "id" in df.columns
        assert "name" in df.columns
        assert "config_method" in df.columns
        assert "best_run_id" in df.columns
        assert "config__internal_key" not in df.columns  # Should be filtered out


def test_export_sweeps_custom_params(runner, mock_wandb_api, mock_sweeps_response, temp_dir):
    """Test export_sweeps command with custom parameters."""
    # Set up the mock API
    api_instance = mock_wandb_api.return_value
    mock_project = mock.MagicMock()
    mock_project.sweeps.return_value = mock_sweeps_response
    api_instance.project.return_value = mock_project
    
    custom_output = "custom-sweeps.csv"
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command with custom parameters
        result = runner.invoke(
            reporting, 
            [
                "sweeps", 
                "--wandb-entity", "custom-entity", 
                "--wandb-project", "custom-project",
                "--output", custom_output
            ]
        )
        
        # Verify the command executed successfully
        assert result.exit_code == 0
        
        # Check API was called correctly
        api_instance.project.assert_called_once_with(name="custom-project", entity="custom-entity")
        
        # Verify custom output file was created
        output_file = Path(custom_output)
        assert output_file.exists()

def test_export_sweeps_no_sweeps(runner, mock_wandb_api, temp_dir):
    """Test export_sweeps command when no sweeps are found."""
    # Set up the mock API to return empty list
    api_instance = mock_wandb_api.return_value
    mock_project = mock.MagicMock()
    mock_project.sweeps.return_value = []
    api_instance.project.return_value = mock_project
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command
        result = runner.invoke(
            reporting, 
            ["sweeps", "--wandb-entity", "test-entity", "--wandb-project", "test-project"]
        )
        
        # Verify the command executed successfully but with warning
        assert result.exit_code == 0
        
        # Verify output file was not created
        output_file = Path("wandb-sweeps.csv")
        assert not output_file.exists()

def test_export_sweeps_best_run_exception(runner, mock_wandb_api, temp_dir):
    """Test export_sweeps command when best_run() raises an exception."""
    # Set up the mock API
    api_instance = mock_wandb_api.return_value
    mock_project = mock.MagicMock()
    
    # Create a sweep that raises an exception on best_run()
    mock_sweep = mock.MagicMock()
    mock_sweep.id = "problem-sweep"
    mock_sweep.name = "problem-sweep-name"
    mock_sweep.config = {"method": "random"}
    mock_sweep.best_run.side_effect = Exception("API error")
    
    mock_project.sweeps.return_value = [mock_sweep]
    api_instance.project.return_value = mock_project
    
    # Change to temporary directory for test
    with runner.isolated_filesystem(temp_dir=temp_dir):
        # Run the command
        result = runner.invoke(
            reporting, 
            ["sweeps", "--wandb-entity", "test-entity", "--wandb-project", "test-project"]
        )
        
        # Verify the command executed successfully
        assert result.exit_code == 0
        
        # Verify output file was created despite the exception
        output_file = Path("wandb-sweeps.csv")
        assert output_file.exists()
        
        # Load and check the CSV content - should have the sweep but no best_run data
        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df.iloc[0]["id"] == "problem-sweep"
        assert "best_run_id" not in df.columns or pd.isna(df.iloc[0].get("best_run_id"))