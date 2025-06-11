import io
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

import torch

log = logging.getLogger(__name__)


@runtime_checkable
class ModelCheckpointer(Protocol):
    """Protocol defining the interface for model checkpointing implementations."""

    @property
    def run_id(self) -> str:
        """Identifier for the current training run."""
        ...

    @property
    def target_metric_name(self) -> str:
        """Name of the target metric for logging."""
        ...

    def save_best(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
    ) -> Path:
        """Save model as the new best version.

        Args:
            model: The model to save
            step_index: Current training epoch
            target_metric: Value of the target metric
            phase: Current training phase

        Returns:
            Path: Location where the model was saved
        """
        ...

    def load_checkpoint(
        self,
        path: Union[Path, str],
    ) -> torch.nn.Module:
        """Load a model checkpoint.

        Args:
            path: Specific checkpoint path to load. If provided, load_best is ignored.

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist
        """
        ...

    def load_best(self, run_id: str) -> torch.nn.Module:
        """Load the best model checkpoint.

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If no best checkpoint exists
        """
        ...

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
        descriptor: Optional[str] = None,
    ) -> Path:
        """Save a prototype update checkpoint.

        Args:
            model: The model to save
            step_index: Current training epoch
            target_metric: Value of the target metric
            phase: Current training phase
            descriptor: Either 'prepush' or 'postpush'

        Returns:
            Path: Location where the model was saved. May not be a file path.
        """
        ...

    def archive_best(self) -> Path:
        """Archive the best run to the artifact directory.

        Returns:
            Path: Location of the archived model.
        """
        ...


@runtime_checkable
class ModelCheckpointerFactory(Protocol):
    def for_run(self, run_id: str, target_metric_name: str) -> ModelCheckpointer:
        """
        Call this factory to create a new checkpointer.
        """
        ...

    def __call__(self, run_id: str, target_metric_name: str) -> ModelCheckpointer:
        """
        Call this factory to create a new checkpointer.
        """
        ...


class RollingFilesystemModelCheckpointer:
    """Handles the mechanics of saving model checkpoints in different formats."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        artifact_dir: Union[str, Path],
        run_id: str,
        target_metric_name: str,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            run_id: Identifier for this training run
            target_metric_name: Name of metric (for logging)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.artifact_dir = Path(artifact_dir) / run_id
        self._run_id = run_id
        self.run_dir = self.checkpoint_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._best_path = None

        self._target_metric_name = target_metric_name

    @property
    def run_id(self) -> str:
        return str(self._run_id)

    @property
    def target_metric_name(self) -> str:
        return str(self._target_metric_name)

    def save_best(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
    ) -> Path:
        """Save model as the new best version."""
        # Save to permanent location
        model_path = self.run_dir / f"{self.run_id}_best.pth"

        # Create descriptive symlink
        metric_str = f"{float(metric):.3g}"
        model_link_path = (
            self.run_dir / f"{self.run_id}_best@{step_index}_{phase}_{metric_str}.pth"
        )

        log.info(
            "Saving model with %s %s to %s, linking to %s.",
            self.target_metric_name,
            metric_str,
            model_path,
            model_link_path,
        )

        # Save model
        torch.save(obj=model, f=str(model_path))

        # Update symlink
        if model_link_path.exists():
            model_link_path.unlink()
        model_link_path.symlink_to(model_path.name)  # Use relative path for symlink
        self._best_path = model_link_path
        return model_path

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
        descriptor: Optional[str] = None,
    ) -> Path:
        """Save a prototype update checkpoint."""
        metric_str = f"{float(metric):.3g}"

        filename_parts = [self.run_id, str(step_index), metric_str]
        if descriptor:
            filename_parts.append(descriptor)

        filename = "_".join(filename_parts) + ".pth"

        run_dir = self.checkpoint_dir / self.run_id
        model_path = run_dir / filename

        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj=model, f=str(model_path))

        log.info("Saved checkpoint to %s", model_path)

        return model_path

    def load_best(self) -> torch.nn.Module:
        """Load the best model checkpoint.

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If no best checkpoint exists
        """
        if not self._best_path:
            raise FileNotFoundError(
                f"No best checkpoint found for run {self.run_id} in {self.checkpoint_dir}"
            )

        log.info("Loading best model from %s", self._best_path)
        return torch.load(str(self._best_path))

    def load_checkpoint(self, path: Union[Path, str]) -> torch.nn.Module:
        """Load a specific checkpoint.

        Args:
            path: Path to the checkpoint to load

        Returns:
            The loaded model

        Raises:
            FileNotFoundError: If the checkpoint doesn't exist
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        log.info("Loading model from %s", checkpoint_path)
        return torch.load(str(checkpoint_path))

    def archive_best(self) -> Path:
        if not self._best_path:
            raise FileNotFoundError("No best checkpoint has been saved yet.")

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.artifact_dir / self._best_path.name
        log.info("Archiving best model to %s", artifact_path)
        shutil.copy(self._best_path, artifact_path)
        return artifact_path


class FilesystemModelCheckpointerFactory:
    """Factory for creating filesystem model checkpointers."""

    def __init__(
        self,
        artifact_dir: Optional[Union[str, Path]] = Path(
            os.environ.get("PPNXT_ARTIFACT_DIR", ".")
        )
        / "runs",
        checkpoint_dir: Optional[Union[str, Path]] = (
            Path(os.environ["PPNXT_CHECKPOINT_DIR"])
            if "PPNXT_CHECKPOINT_DIR" in os.environ
            else None
        ),
    ):
        self.artifact_dir = Path(artifact_dir)
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = self.artifact_dir / "checkpoints"

    def for_run(
        self, run_id: str, target_metric_name: str
    ) -> RollingFilesystemModelCheckpointer:
        return RollingFilesystemModelCheckpointer(
            checkpoint_dir=self.checkpoint_dir,
            artifact_dir=self.artifact_dir,
            run_id=run_id,
            target_metric_name=target_metric_name,
        )

    def __call__(self, run_id: str, target_metric_name: str):
        return self.for_run(run_id, target_metric_name)


class BestOnlyInMemoryModelCheckpointer:
    """
    A model checkpointer that saves the best model in memory and drops all other models.
    """

    def __init__(self, run_id: str, target_metric_name: str):
        self.best_model_stream = None
        self.best_metadata = None
        self._run_id = run_id
        self._target_metric_name = target_metric_name

    @property
    def run_id(self) -> str:
        return str(self._run_id)

    @property
    def target_metric_name(self) -> str:
        return str(self._target_metric_name)

    @run_id.setter
    def run_id(self, value: str):
        self._run_id = value

    def save_best(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
    ) -> Path:
        log.info(
            "Saving new best model with target_metric=%.4f at step %d during phase '%s'",
            self.target_metric_name,
            step_index,
            phase,
        )
        self.best_model_stream = io.BytesIO()
        torch.save(model.state_dict(), self.best_model_stream)
        self.best_model_stream.seek(0)  # Reset the stream position
        self.best_metadata = {
            "step_index": step_index,
            "target_metric": self.target_metric_name,
            "phase": phase,
        }
        return "memory://best_model"

    def load_checkpoint(
        self,
        path: Union[str, Path],
    ) -> torch.nn.Module:
        log.info("load_checkpoint called with path: %s", path)
        if path == "memory://best_model":
            return self.load_best()
        raise FileNotFoundError(
            "In-memory checkpointer does not save regular checkpoints."
        )

    def load_best(self) -> torch.nn.Module:
        if self.best_model_stream is None:
            raise FileNotFoundError("No best model has been saved yet.")
        self.best_model_stream.seek(0)  # Reset the stream position
        model = torch.nn.Module()  # Create a new model instance
        model.load_state_dict(torch.load(self.best_model_stream))
        return model

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step_index: int,
        metric: float,
        phase: str,
        descriptor: Optional[str] = None,
    ) -> Path:
        log.info(
            "save_proto_checkpoint called with target_metric=%.4f, epoch_index=%d, stage='%s'",
            self.target_metric_name,
            step_index,
            phase,
        )
        return f"memory://proto_checkpoint/{phase}"

    def archive_best(self) -> Path:
        pass
