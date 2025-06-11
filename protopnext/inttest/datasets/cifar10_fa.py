import logging
import os
import pathlib
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader

from protopnet.datasets.cifar10 import CIFARSplitDataloaders
from protopnet.datasets.torch_extensions import (
    CombinedDataloader,
    SingleSamplePerClassDataloader,
)

log = logging.getLogger(__name__)


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("CIFAR10_DIR", "CIFAR10"),
    train_dir: str = "train",
    val_dir: str = "val",
    project_dir: str = "project",
    image_size=(32, 32),
    batch_sizes={"train": 1000, "project": 1000, "val": 1000},
):
    split_dataloaders = CIFARSplitDataloaders.create(data_path, image_size, batch_sizes)

    train_loader = split_dataloaders.train_loader
    train_loader = SparseFineAnnoDataloader(train_loader, image_size)
    train_aux_loader = SingleSamplePerClassDataloader(train_loader)
    train_aux_loader = FakeFineAnnoDataloader(train_aux_loader, image_size)
    split_dataloaders.train_loader = CombinedDataloader(
        [train_loader, train_aux_loader], sparse_keys=["fine_annotation"]
    )

    val_loader = SparseFineAnnoDataloader(split_dataloaders.val_loader, image_size)
    split_dataloaders.val_loader = val_loader

    return split_dataloaders


class SparseFineAnnoDataloader(DataLoader):
    def __init__(self, dataloader: DataLoader, fa_size: Tuple[int, int]):
        self.dataloader = dataloader
        self.fa_size = fa_size
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        for batch in self.dataloader:
            fine_annotation = torch.sparse_coo_tensor(
                size=(batch["img"].shape[0], 1, *self.fa_size)
            )
            batch["fine_annotation"] = fine_annotation

            yield batch

    def __len__(self) -> int:
        return self.batch_size


class FakeFineAnnoDataloader(DataLoader):
    def __init__(self, dataloader: DataLoader, fa_size: Tuple[int, int]):
        self.dataloader = dataloader
        self.fa_size = fa_size
        self.batch_size = dataloader.batch_size

    def __iter__(self):
        for batch in self.dataloader:
            fine_annotation = torch.zeros(batch["img"].shape[0], 1, *self.fa_size)
            fine_annotation[:, :, : self.fa_size[0] // 2] = 1
            batch["fine_annotation"] = fine_annotation

            yield batch

    def __len__(self) -> int:
        return self.batch_size
