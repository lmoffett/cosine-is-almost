import os
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from protopnet.datasets import cars_cropped, cub200_cropped
from protopnet.datasets.dataset_prep import create_splits


@pytest.fixture
def mock_cropped_cars(mock_cars):
    for dir in ["images_cropped", "train_cropped", "val_cropped"]:
        cropped_dir = mock_cars / dir
        if cropped_dir.exists():
            shutil.rmtree(cropped_dir)
    assert cropped_dir.exists() == False

    cars_cropped.crop_cars(mock_cars)

    create_splits(
        base_dir=mock_cars,
        image_dir="images_cropped",
        train_dir="train_cropped",
        val_dir="val_cropped",
        val_ratio=0.7,
    )
    return mock_cars


def test_train_dataloaders(mock_cropped_cars):
    splits_dataloaders = cars_cropped.train_dataloaders(
        data_path=mock_cropped_cars, part_labels=False
    )

    assert splits_dataloaders.train_loader.batch_size == 95
    assert splits_dataloaders.val_loader.batch_size == 100
    assert splits_dataloaders.project_loader.batch_size == 75
    assert splits_dataloaders.test_loader.batch_size == 100
    assert splits_dataloaders.num_classes == 196

    assert len(splits_dataloaders.train_loader.dataset) == 2
    assert len(splits_dataloaders.val_loader.dataset) == 1
    assert len(splits_dataloaders.project_loader.dataset) == 2
    assert len(splits_dataloaders.test_loader.dataset) == 2


# Test function to check if images are cropped correctly
def test_crop_cars(mock_cropped_cars):
    cropped_images_dir = mock_cropped_cars / "images_cropped"
    assert cropped_images_dir.exists()

    # Check if cropped images exist
    cropped_image1 = cropped_images_dir / "0" / "0_000001.jpg"
    cropped_image2 = cropped_images_dir / "1" / "1_000002.jpg"
    cropped_image5 = cropped_images_dir / "1" / "1_100005.jpg"

    assert cropped_image1.exists()
    assert cropped_image2.exists()
    assert cropped_image5.exists()

    # Check if the dimensions of the cropped images are as expected
    im1 = Image.open(cropped_image1)
    im2 = Image.open(cropped_image2)
    im5 = Image.open(cropped_image5)

    assert im1.size == (216, 95)
    assert im2.size == (476, 184)
    assert im5.size == (476, 184)
