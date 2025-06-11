import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Union

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..datasets.dataset_prep import calculate_split_mean_std
from .torch_extensions import VisionDictMixin, seed_worker_for_reproducability

log = logging.getLogger(__name__)


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("CIFAR10_DIR", "CIFAR10"),
    train_dir: str = "train",
    val_dir: str = "val",
    project_dir: str = "project",
    image_size=(32, 32),
    batch_sizes={"train": 1000, "project": 1000, "val": 1000},
):
    return CIFARSplitDataloaders.create(data_path, image_size, batch_sizes)


class DictCIFAR10(VisionDictMixin, datasets.CIFAR10):
    """
    CIFAR10 Implementation that returns dicts with sample_ids
    """

    pass


@dataclass
class CIFARSplitDataloaders:
    data_path: str
    num_classes: int
    image_size: tuple
    batch_sizes: dict
    train_loader: DataLoader
    val_loader: DataLoader
    project_loader: DataLoader
    test_loader: DataLoader

    @staticmethod
    def create(data_path, image_size, batch_sizes):
        seed = int(os.environ.get("PPNXT_SEED", 1234))
        generator = torch.Generator()
        generator.manual_seed(seed)

        # Split the training dataset into training and validation sets
        distribution_dataset_raw = DictCIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(size=(image_size[0], image_size[1])),
                    transforms.ToTensor(),
                ]
            ),
        )
        # 80% training, 20% validation
        train_size = int(0.8 * len(distribution_dataset_raw))
        val_size = len(distribution_dataset_raw) - train_size

        # same as train, but without transforms
        distribution_dataset, _ = random_split(
            distribution_dataset_raw,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1234),
        )

        train_distribution = calculate_split_mean_std(distribution_dataset)
        normalize = transforms.Normalize(
            mean=train_distribution.mean, std=train_distribution.std
        )

        train_transform = transforms.Compose(
            [
                transforms.RandomChoice(
                    [
                        transforms.RandomRotation(degrees=15),  # rotation
                        transforms.RandomPerspective(
                            distortion_scale=0.2
                        ),  # perspective skew
                        transforms.RandomAffine(degrees=0, shear=10),  # shear
                    ]
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(size=(image_size[0], image_size[1])),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Load CIFAR-10 dataset
        train_dataset = DictCIFAR10(
            root=data_path, train=True, download=True, transform=train_transform
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(size=(image_size[0], image_size[1])),
                transforms.ToTensor(),
                normalize,
            ]
        )

        eval_dataset = DictCIFAR10(
            root=data_path, train=True, download=True, transform=eval_transform
        )

        project_dataset, val_dataset = random_split(
            eval_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1234),
        )
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(1234),
        )

        # Create DataLoaders for training, validation, and test sets
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_sizes["train"],
            shuffle=True,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )
        project_loader = DataLoader(
            project_dataset,
            batch_size=batch_sizes["project"],
            shuffle=False,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_sizes["val"],
            shuffle=False,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )

        # Add this after creating the val_dataset
        test_dataset = DictCIFAR10(
            root=data_path,
            train=False,  # Use the test split
            download=True,
            transform=eval_transform,  # Use the same transform as validation
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=(
                batch_sizes["val"] if "test" not in batch_sizes else batch_sizes["test"]
            ),
            shuffle=False,
            generator=generator,
            worker_init_fn=seed_worker_for_reproducability,
        )

        return CIFARSplitDataloaders(
            data_path=data_path,
            num_classes=10,  # CIFAR-10 has 10 classes
            image_size=image_size,
            batch_sizes=batch_sizes,
            train_loader=train_loader,
            val_loader=val_loader,
            project_loader=project_loader,
            test_loader=test_loader,
        )
