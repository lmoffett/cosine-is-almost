import csv
import datetime
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from protopnet.eval import ModelParameter, run


class TestModelParameter:
    def setup_method(self):
        """Setup for each test method"""
        self.param_type = ModelParameter()
        self.runner = CliRunner()

    @pytest.mark.parametrize("file_ext", [".pth", ".pt", ".ckpt", ".model", ".bin"])
    def test_supported_file_types(self, file_ext):
        """Test converting files of a single model"""
        result = self.param_type.convert(f"model{file_ext}", None, None)
        assert result == {"model": f"model{file_ext}"}

    def test_convert_csv_file(self, temp_dir):
        """Test converting a .csv file path with mocked file reading"""
        csv_content = "model1,path/to/model1.pth\nmodel2,path/to/model2.pth"

        temp_file_path = temp_dir / "models.csv"
        with open(temp_file_path, "w") as f:
            f.write(csv_content)

        try:
            result = self.param_type.convert(temp_file_path, None, None)
            assert result == {
                "model1": "path/to/model1.pth",
                "model2": "path/to/model2.pth",
            }
        finally:
            os.unlink(temp_file_path)

    def test_parse_model_csv_with_comments_and_empty_lines(self, temp_dir):
        """Test parsing a CSV file with comments and empty lines"""
        csv_content = """# This is a comment
model1,path/to/model1.pth

# Another comment
model2,path/to/model2.pth
,
empty_value,
,empty_path
"""
        temp_file_path = temp_dir / "models.csv"
        with open(temp_file_path, "w") as temp_file:
            temp_file.write(csv_content)
        try:
            result = self.param_type.parse_model_csv(temp_file_path)
            assert result == {
                "model1": "path/to/model1.pth",
                "model2": "path/to/model2.pth",
            }
        finally:
            os.unlink(temp_file_path)  # Clean up the temp file

    def test_unsupported_file_type(self):
        """Test converting an unsupported file type"""
        with pytest.raises(click.BadParameter) as excinfo:
            self.param_type.convert("model.txt", None, None)

        assert "Unsupported file type" in str(excinfo.value)

    def test_csv_file_not_found(self):
        """Test handling a CSV file that doesn't exist"""
        with pytest.raises(click.BadParameter) as excinfo:
            self.param_type.convert("nonexistent.csv", None, None)

        assert "Failed to parse CSV file" in str(excinfo.value)

    def test_invalid_csv_format(self):
        """Test handling an invalid CSV format"""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w+", delete=False
        ) as temp_file:
            temp_file.write("this is not a valid CSV format")
            temp_file_path = temp_file.name

        try:
            # This should still work because our parser is flexible
            result = self.param_type.convert(temp_file_path, None, None)
            assert result == {}  # No valid entries
        finally:
            os.unlink(temp_file_path)


@pytest.fixture
def mock_load_and_eval():
    """Mock the load_and_eval function to return a predictable dataframe"""
    with patch("protopnet.eval.load_and_eval") as mock:
        # Create a simple dataframe as the return value
        df = pd.DataFrame(
            {
                "model": ["model1", "model2"],
                "accuracy": [0.95, 0.97],
                "precision": [0.94, 0.96],
                "recall": [0.93, 0.95],
            }
        )
        mock.return_value = df
        yield mock


@pytest.fixture
def mock_training_loaders():
    """Mock the datasets module"""
    with patch("protopnet.datasets.training_dataloaders") as mock:
        mock_dataloaders = MagicMock()
        mock_dataloaders.num_classes = 10
        mock_dataloaders.val_loader = MagicMock()
        mock_dataloaders.test_loader = MagicMock()

        mock.return_value = mock_dataloaders
        yield mock


@pytest.fixture
def model_file(temp_dir):
    """Create a mock model file"""
    model_path = temp_dir / "test_model.pth"
    # Create an empty file
    model_path.touch()
    return str(model_path)


@pytest.fixture
def models_csv(temp_dir):
    """Create a CSV file with model paths"""
    csv_path = temp_dir / "models.csv"

    # Write CSV with model identifiers and paths
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model", "path"])
        writer.writerow(["model1", "/path/to/model1.pth"])
        writer.writerow(["model2", "/path/to/model2.pth"])

    return str(csv_path)


@pytest.fixture(params=["single_model", "multiple_models"])
def model_input(request, model_file, models_csv):
    """Parameterized fixture for different model input types"""
    if request.param == "single_model":
        return model_file
    else:
        return models_csv


def test_run_minimal_args(
    model_input, mock_load_and_eval, mock_training_loaders, temp_dir
):
    """Test the run command with only the required arguments"""
    runner = CliRunner()

    result = runner.invoke(
        run,
        [
            model_input,
        ],
    )

    # Check that the command executed without errors
    assert result.exit_code == 0

    mock_load_and_eval.assert_called_once()

    mock_training_loaders.assert_called_once()


def test_run_all_args(model_input, mock_load_and_eval, mock_training_loaders, temp_dir):
    """Test the run command with all arguments specified"""
    runner = CliRunner()
    output_file = temp_dir / "output.pkl"

    # Run the Click command with all arguments
    result = runner.invoke(
        run,
        [
            model_input,
            "--output-file",
            str(output_file),
            "--device",
            "cpu",
            "--batch-size",
            "50",
            "--dataset",
            "cifar10",
            "--acc-only",
        ],
    )

    assert result.exit_code == 0

    mock_load_and_eval.assert_called_once()
    _, kwargs = mock_load_and_eval.call_args
    assert kwargs.get("acc_only", False) is True
    assert kwargs.get("device", None) == torch.device("cpu")

    mock_training_loaders.assert_called_once_with(
        "CIFAR10", batch_sizes={"train": 1, "project": 1, "val": 50, "test": 50}
    )

    assert output_file.exists(), "Output file link should exist"


@patch("protopnet.eval.datetime")
def test_output_file_naming(
    mock_datetime, model_input, mock_load_and_eval, mock_training_loaders, temp_dir
):
    """Test the naming convention for output files"""
    # Mock the datetime to have a fixed return value
    mock_fixed_datetime = MagicMock()
    mock_fixed_datetime.now.return_value = datetime.datetime(2023, 1, 1, 12, 0, 0)
    mock_datetime.datetime = mock_fixed_datetime

    runner = CliRunner()
    output_file = temp_dir / "custom_output.pkl"

    # Run the command with the output file specified
    result = runner.invoke(
        run,
        [
            model_input,
            "--output-file",
            str(output_file),
        ],
    )

    # Check that the command executed without errors
    assert result.exit_code == 0

    expected_file = (
        temp_dir / f"custom_output_{datetime.datetime(2023, 1, 1, 12, 0, 0)}.pkl"
    )
    assert expected_file.name in [
        p.name for p in temp_dir.iterdir()
    ], f"Expected file {expected_file.name} not found in {[p.name for p in temp_dir.iterdir()]}"

    assert output_file.exists(), "Output file link should exist"


def test_model_parameter_conversion(
    model_input, mock_load_and_eval, mock_training_loaders, temp_dir
):
    """Test that the ModelParameter custom type correctly converts the input"""
    runner = CliRunner()

    # Run the command
    result = runner.invoke(
        run,
        [
            model_input,
        ],
    )

    # Check that the command executed without errors
    assert result.exit_code == 0

    # Get the dictionary that was passed to load_and_eval
    args, _ = mock_load_and_eval.call_args
    model_dict = args[0]

    # Check that model_dict is a dictionary
    assert isinstance(
        model_dict, dict
    ), "The model parameter should be converted to a dictionary"

    # If single model file, it should have a single entry
    # If models CSV, it should have multiple entries from the CSV
    if "models.csv" in model_input:
        assert len(model_dict) >= 2, "For CSV input, should have at least two models"
    else:
        assert model_dict == {
            "model": model_input
        }, "For single model file, should have one entry"
