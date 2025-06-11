from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

from protopnet.train.checkpointing import RollingFilesystemModelCheckpointer


# Simple model for testing
class TinyModel(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([value]))

    def forward(self, x):
        return x + self.param


@pytest.fixture
def checkpointer(temp_dir: Path) -> RollingFilesystemModelCheckpointer:
    """Creates a checkpointer instance for testing."""
    return RollingFilesystemModelCheckpointer(
        checkpoint_dir=temp_dir / "checkpointing",
        artifact_dir=temp_dir / "artifacts",
        run_id="test_run",
        target_metric_name="test_metric",
    )


@pytest.fixture
def model() -> TinyModel:
    """Creates a simple model for testing."""
    return TinyModel(value=12.0)


def test_save_best_creates_files(
    checkpointer: RollingFilesystemModelCheckpointer, model: TinyModel, temp_dir: Path
):
    """Test that save_best creates both the permanent file and symlink."""
    path = checkpointer.save_best(
        model=model, step_index=10, metric=0.95, phase="train"
    )

    assert path.exists()

    expected_link: Path = (
        temp_dir / "checkpointing" / "test_run" / "test_run_best@10_train_0.95.pth"
    )
    assert expected_link.exists()
    assert expected_link.is_symlink()
    assert path.resolve() == expected_link.resolve()

    archive_path = checkpointer.archive_best()

    expected_best_link: Path = (
        temp_dir / "artifacts" / "test_run" / "test_run_best@10_train_0.95.pth"
    )
    assert archive_path.exists()
    assert archive_path.resolve() == expected_best_link.resolve()


@pytest.mark.parametrize("descriptor", [None, "prepush"])
def test_save_proto_checkpoint(
    checkpointer: RollingFilesystemModelCheckpointer, model: TinyModel, descriptor: str
):
    """Test saving prototype checkpoints with and without descriptor."""
    path = checkpointer.save_checkpoint(
        model=model,
        step_index=5,
        metric=0.85,
        phase="train",
        descriptor=descriptor,
    )
    assert path.exists()
    descriptor = f"_{descriptor}" if descriptor else ""
    assert path.name == f"test_run_5_0.85{descriptor}.pth"


def test_load_best_loads_correct_model(
    checkpointer: RollingFilesystemModelCheckpointer, model: TinyModel
):
    """Test that load_best loads the model with correct parameters."""
    # Save the model first
    checkpointer.save_best(model=model, step_index=1, metric=0.9, phase="train")

    # Load the model
    loaded_model = checkpointer.load_best()
    assert isinstance(loaded_model, TinyModel)
    assert torch.allclose(loaded_model.param, model.param)


def test_load_checkpoint(
    checkpointer: RollingFilesystemModelCheckpointer,
    model: TinyModel,
):
    """Test loading a specific checkpoint."""
    # Save a proto checkpoint
    path = checkpointer.save_checkpoint(
        model=model,
        step_index=1,
        metric=0.9,
        phase="train",
        descriptor="test",
    )

    # Load the checkpoint
    loaded_model = checkpointer.load_checkpoint(path)
    assert isinstance(loaded_model, TinyModel)
    assert torch.allclose(loaded_model.param, model.param)


def test_load_best_missing_raises(temp_dir: Path):
    """Test that attempting to load a non-existent best model raises FileNotFoundError."""

    no_best_checkpointer = RollingFilesystemModelCheckpointer(
        checkpoint_dir=temp_dir / "checkpointing",
        artifact_dir=temp_dir / "artifacts",
        run_id="no_best",
        target_metric_name="test_metric",
    )
    with pytest.raises(FileNotFoundError):
        no_best_checkpointer.load_best()

    with pytest.raises(FileNotFoundError):
        no_best_checkpointer.archive_best()


def test_load_checkpoint_missing_raises(
    checkpointer: RollingFilesystemModelCheckpointer, temp_dir: Path
):
    """Test that attempting to load a non-existent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        checkpointer.load_checkpoint(temp_dir / "nonexistent.pth")
