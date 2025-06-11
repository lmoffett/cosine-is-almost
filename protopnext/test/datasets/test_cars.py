import os

import pandas as pd

from protopnet.datasets import cars


def test_output_files(mock_cars):
    images_path = os.path.join(mock_cars, "images.txt")
    assert os.path.exists(images_path)
    images_df = pd.read_csv(images_path, sep=" ", header=None)
    assert images_df.shape == (5, 2)

    bounding_boxes_path = os.path.join(mock_cars, "bounding_boxes.txt")
    assert os.path.exists(bounding_boxes_path)
    bounding_boxes_df = pd.read_csv(bounding_boxes_path, sep=" ", header=None)
    assert bounding_boxes_df.shape == (5, 5)

    train_test_split_path = os.path.join(mock_cars, "train_test_split.txt")
    assert os.path.exists(train_test_split_path)
    train_test_split_df = pd.read_csv(train_test_split_path, sep=" ", header=None)
    assert train_test_split_df.shape == (5, 2)

    image_class_labels_path = os.path.join(mock_cars, "image_class_labels.txt")
    assert os.path.exists(image_class_labels_path)
    image_class_labels_df = pd.read_csv(image_class_labels_path, sep=" ", header=None)
    assert image_class_labels_df.shape == (5, 2)


def test_train_dataloaders(mock_cars):
    splits_dataloaders = cars.train_dataloaders(data_path=mock_cars, part_labels=False)

    assert splits_dataloaders.train_loader.batch_size == 95
    assert splits_dataloaders.val_loader.batch_size == 100
    assert splits_dataloaders.project_loader.batch_size == 75
    assert splits_dataloaders.test_loader.batch_size == 100
    assert splits_dataloaders.num_classes == 196

    assert len(splits_dataloaders.train_loader.dataset) == 2
    assert len(splits_dataloaders.val_loader.dataset) == 1
    assert len(splits_dataloaders.project_loader.dataset) == 2
    assert len(splits_dataloaders.test_loader.dataset) == 2
