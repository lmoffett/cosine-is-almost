import os
import unittest
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import Dataset

from protopnet.datasets import dataset_prep
from protopnet.datasets.dataset_prep import calculate_split_mean_std


class MockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            "img": self.data[index],
            "target": self.labels[index],
            "sample_id": index,
        }


@patch("builtins.open", new_callable=unittest.mock.mock_open)
def test_create_splits(mock_open, mock_cub200):
    base_dir, class_dir_name, images_txt_content = mock_cub200

    train_dir = base_dir / "train"
    test_dir = base_dir / "test"
    val_dir = base_dir / "validation"

    split_txt_content = "1 1\n2 0\n3 1"

    # Mock the file contents
    mock_open.side_effect = [
        unittest.mock.mock_open(read_data=images_txt_content).return_value,
        unittest.mock.mock_open(read_data=split_txt_content).return_value,
    ]

    # Call the function with the simulated directory
    dataset_prep.create_splits(base_dir, val_ratio=0.5)

    assert os.listdir(train_dir / class_dir_name) == [
        "Black_Footed_Albatross_0003_796113.jpg"
    ], "Only one file should remain in train."
    assert os.listdir(val_dir / class_dir_name) == [
        "Black_Footed_Albatross_0001_796111.jpg"
    ], "Only one file should be moved to validation."
    assert os.listdir(test_dir / class_dir_name) == [
        "Black_Footed_Albatross_0002_796112.jpg"
    ], "No files should be in the test directory."


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 9])
def test_calculate_split_mean_std_correct_values(batch_size):
    """Test if the function calculates correct mean and std for a simple dataset."""

    images = torch.arange(0, 512).float().reshape(8, 4, 4, 4) / 255.0
    labels = torch.zeros(8)
    dataset = MockDataset(images, labels)

    # Call the function
    result = calculate_split_mean_std(dataset, batch_size=batch_size, num_workers=0)

    # Expected mean and std
    expected_mean = images.mean(dim=(0, 2, 3))
    expected_std = images.std(dim=(0, 2, 3), unbiased=False)

    # Assert values are correct
    assert torch.allclose(
        result.mean, expected_mean
    ), f"Expected: {expected_mean}, got: {result.mean}"
    assert torch.allclose(
        result.std, expected_std
    ), f"Expected: {expected_std}, got: {result.std}"
