import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

SPLIT_SEED = 1234


@dataclass
class SplitDistribution:
    """
    Dataclass to store mean and standard deviation of a dataset split.
    """

    mean: float
    std: float


# For dogs, set images_dir_name="Images", cast_ids_to_int=False
def create_splits(
    base_dir,
    image_dir="images",
    train_dir="train",
    val_dir="validation",
    test_dir="test",
    val_ratio=0.1,
    seed=SPLIT_SEED,
    cast_ids_to_int=True,
):
    # Define the paths
    images_dir = Path(base_dir) / image_dir
    split_file = Path(base_dir) / "train_test_split.txt"
    images_file = Path(base_dir) / "images.txt"

    # Read image names and IDs
    log.info("Reading image names and IDs")
    image_paths = {}
    with open(images_file, "r") as f:
        for line in f:
            image_id, image_path = line.strip().split(" ")
            if cast_ids_to_int:
                image_paths[int(image_id)] = image_path
            else:
                image_paths[image_id] = image_path

    # Read train/test split
    log.info("Reading test Splits")
    train_test_split = {}
    with open(split_file, "r") as f:
        for line in f:
            image_id, is_train = line.strip().split(" ")
            if cast_ids_to_int:
                train_test_split[int(image_id)] = int(is_train)
            else:
                train_test_split[image_id] = int(is_train)

    # Create directories
    log.info("Creating train, test, validation directories in %s", base_dir)
    train_dir = Path(base_dir) / train_dir
    test_dir = Path(base_dir) / test_dir
    val_dir = Path(base_dir) / val_dir
    dirs = [train_dir, test_dir, val_dir]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Split into train, test, validation
    train_files = {}
    log.info("Creating train-test split")
    for image_id, path in tqdm.tqdm(image_paths.items()):
        target_dir = test_dir if train_test_split[image_id] == 0 else train_dir
        source_path = images_dir / path
        class_dir, image_file = path.split("/")[-2:]
        (target_dir / class_dir).mkdir(parents=True, exist_ok=True)
        link_path = target_dir / class_dir / image_file

        # Create symlink
        if not link_path.exists():
            link_path.symlink_to(
                Path("../..")
                / image_dir
                / source_path.relative_to(source_path.parent.parent)
            )

        if class_dir not in train_files:
            train_files[class_dir] = []

        if target_dir == train_dir:
            train_files[class_dir].append((source_path, link_path))

    # Create validation split from train set (10%)
    validation_files = []
    for class_dir, files in train_files.items():
        class_files_for_validation = list(files)
        random.shuffle(class_files_for_validation, random=random.Random(seed).random)

        val_count = int(len(files) * val_ratio)
        validation_files.extend(class_files_for_validation[:val_count])

    log.info("Splitting validation set of size %d from train", val_count)
    # Move selected train files to validation
    for source_path, link_path in tqdm.tqdm(validation_files):
        class_dir, image_file = link_path.parts[-2:]
        (val_dir / class_dir).mkdir(parents=True, exist_ok=True)

        val_link_path = val_dir / class_dir / image_file
        if not val_link_path.exists():
            val_link_path.symlink_to(
                Path("../..")
                / image_dir
                / source_path.relative_to(source_path.parent.parent)
            )
        os.remove(link_path)


def calculate_split_mean_std(
    dataset: torch.utils.data.Dataset, batch_size=64, num_workers=2
):
    """
    Calculate the mean and standard deviation for a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to calculate statistics for.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of workers for the DataLoader.

    Returns:
        tuple: Mean and standard deviation as tensors.
    """
    # DataLoader for the dataset
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Initialize mean and std tensors
    batch_means = []
    batch_vars = []
    batch_samples = []

    logging.info("Calculating mean and standard deviation on split...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Iterate over DataLoader
    for batch in tqdm.tqdm(loader):
        # Reshape images to flatten spatial dimensions
        img = batch["img"].to(device)
        images = img.view(img.size(0), img.size(1), -1)

        batch_samples.append(images.size(0))
        batch_means.append(images.mean(dim=(0, 2)).detach())
        batch_vars.append(images.var(dim=(0, 2), unbiased=False).detach())
        del img, images

    sample_tensor = torch.tensor(batch_samples, dtype=torch.float32, device=device)
    total_samples = sample_tensor.sum()
    batch_means_tensor = torch.stack(batch_means).T
    channel_means = (batch_means_tensor @ sample_tensor) / total_samples

    # Calculate overall variance
    # Weighted sum of intra-batch variances
    batch_vars_tensor = torch.stack(batch_vars).T
    intra_batch_variance = (batch_vars_tensor @ sample_tensor) / total_samples

    # Contribution of batch mean differences
    mean_diff_contrib = (
        ((batch_means_tensor - channel_means.unsqueeze(1)) ** 2) @ sample_tensor
    ) / total_samples

    # Final variance
    channel_vars = intra_batch_variance + mean_diff_contrib
    channel_stds = torch.sqrt(channel_vars)

    distribution = SplitDistribution(
        mean=channel_means.detach().cpu(), std=channel_stds.detach().cpu()
    )
    logging.info(f"Image distribution is {distribution}")
    return distribution
