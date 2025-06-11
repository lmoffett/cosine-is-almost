from typing import Protocol, Union

import torch

from .checkpointing import ModelCheckpointer
from .logging.types import TrainLogger


class ProtoPNetTrainer(Protocol):
    @property
    def device(self) -> Union[str, torch.device]:
        """The device (e.g., 'cuda' or 'cpu') where training occurs."""
        ...

    @property
    def metric_log(self) -> TrainLogger:
        """A dictionary storing training logs."""
        ...

    @property
    def checkpointer(self) -> ModelCheckpointer:
        """An object responsible for saving and loading model checkpoints."""
        ...
