import os
import pathlib
from typing import Union

from .torch_extensions import NPSplitDataloaders


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("MAMMO_DIR", "MAMMO"),
    train_dir: str = "train",
    push_dir: str = "push",
    val_dir: str = "validation",
    train_aux_dir: str = "train_auxiliary",
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    with_fa=True,
    with_aux=True,
    combine_datasets_method="continue",
):
    """
    Creates train, push, val dataloaders for mammo data
    Loads .npy data (does not have cached part labels)
    Does not manually augment train like other datasets as it is already done
    """
    if with_fa and "train_auxiliary" not in batch_sizes:
        batch_sizes["train_auxiliary"] = 10

    return NPSplitDataloaders(
        data_path=data_path,
        image_size=image_size,
        batch_sizes=batch_sizes,
        train_dir=train_dir,
        project_dir=push_dir,
        val_dir=val_dir,
        train_aux_dir=train_aux_dir,
        with_fa=with_fa,
        with_aux=with_aux,
        combine_datasets_method=combine_datasets_method,
    )
