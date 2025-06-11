import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy
from PIL import Image

from protopnet.datasets import cars
from protopnet.datasets.dataset_prep import create_splits


@pytest.fixture(scope="package")
def mock_cub200(temp_root_dir):
    # Setup base directory and paths
    base_dir = Path(temp_root_dir / "test" / "datasets" / "cub200")

    class_dir_name = "001.Black_footed_Albatross"
    class_dir = base_dir / "images" / class_dir_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # Mock the file reading operations
    images_txt_content = f"""1 {class_dir_name}/Black_Footed_Albatross_0001_796111.jpg
2 {class_dir_name}/Black_Footed_Albatross_0002_796112.jpg
3 {class_dir_name}/Black_Footed_Albatross_0003_796113.jpg"""

    with open(base_dir / "images.txt", "w") as f:
        f.write(images_txt_content)

    for row in images_txt_content.split("\n"):
        file = row.split(" ")[-1]
        image_path = base_dir / "images" / file
        image_path.touch()

    yield base_dir, class_dir_name, images_txt_content

    shutil.rmtree(base_dir)


@pytest.fixture(scope="package")
def mock_cars(temp_root_dir):
    cars_root = temp_root_dir / "cars"
    train_images_dir = cars_root / "cars_train" / "cars_train"
    test_images_dir = cars_root / "cars_test" / "cars_test"

    train_images_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(parents=True, exist_ok=True)

    # Create blank train images
    for image_name in ["00001.jpg", "00002.jpg", "00003.jpg"]:
        image_path = train_images_dir / image_name
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path)

    # Create blank test images
    for image_name in ["00004.jpg", "00005.jpg"]:
        image_path = test_images_dir / image_name
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path)

    train_data = pd.DataFrame(
        {
            "x1": [30, 100, 27],
            "y1": [52, 19, 27],
            "x2": [246, 576, 57],
            "y2": [147, 203, 57],
            "Class": [0, 1, 1],
            "image": ["00001.jpg", "00002.jpg", "00003.jpg"],
        }
    )
    train_data.to_csv(cars_root / "cardatasettrain.csv", index=False)

    test_data = pd.DataFrame(
        {
            "x1": [50, 110],
            "y1": [60, 29],
            "x2": [256, 586],
            "y2": [157, 213],
            "image": ["00004.jpg", "00005.jpg"],
        }
    )
    test_data.to_csv(cars_root / "cardatasettest.csv", index=False)

    cars_annos = {
        "annotations": np.expand_dims(
            np.array(
                [
                    tuple(["00004.jpg", 50, 60, 256, 157, 0, 1]),
                    tuple(["00005.jpg", 110, 29, 586, 213, 1, 1]),
                ],
                dtype=object,
            ),
            0,
        )
    }
    scipy.io.savemat(cars_root / "cars_annos.mat", cars_annos)

    cars.parse_cars_metadata(root_dir=str(cars_root.resolve()))
    create_splits(base_dir=cars_root, image_dir="car_ims", val_ratio=0.7)

    return cars_root
