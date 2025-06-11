import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import pandas as pd
import scipy.io
from PIL import Image
from tqdm import tqdm

from . import torch_extensions
from .torch_extensions import CachedPartLabels

log = logging.getLogger(__name__)


def parse_dogs_metadata(root_dir=os.environ.get("DOGS_DIR", "DOGS")):
    source_dir = os.path.join(root_dir, "Annotation")

    # First, convert XML style bounding boxes to a text file
    target_file = os.path.join(root_dir, "bounding_boxes.txt")
    with open(target_file, "w") as f:
        for class_dir in os.listdir(source_dir):
            for filename in os.listdir(os.path.join(source_dir, class_dir)):
                data = ET.parse(os.path.join(source_dir, class_dir, filename))
                cur_object = data.find("object")
                box = cur_object.find("bndbox")
                new_row = f'{filename} {float(box.find("xmin").text)} {float(box.find("ymin").text)} {float(box.find("xmax").text)} {float(box.find("ymax").text)}\n'
                f.write(new_row)

    matlab_file = scipy.io.loadmat(os.path.join(root_dir, "file_list.mat"))
    train_labels = scipy.io.loadmat(os.path.join(root_dir, "train_list.mat"))

    train_imgs = [t[0][0] for t in train_labels["file_list"]]

    description_df = pd.DataFrame()
    description_df["labels"] = matlab_file["labels"].flatten()
    description_df["annotation_list"] = matlab_file["annotation_list"].flatten()
    description_df["file_list"] = matlab_file["file_list"].flatten()

    def anno_to_str(row):
        return row["annotation_list"][0]

    description_df["annotation_list"] = description_df.apply(anno_to_str, axis=1)

    def file_to_str(row):
        return row["file_list"][0]

    description_df["file_list"] = description_df.apply(file_to_str, axis=1)

    def get_img_id(row):
        return row["annotation_list"].split("/")[-1]

    description_df["img_id"] = description_df.apply(get_img_id, axis=1)

    def is_train(row):
        return 1 if row["file_list"] in train_imgs else 0

    description_df["is_train"] = description_df.apply(is_train, axis=1)

    description_df[["img_id", "file_list"]].to_csv(
        os.path.join(root_dir, "images.txt"), header=None, index=False, sep=" "
    )
    description_df[["img_id", "labels"]].to_csv(
        os.path.join(root_dir, "image_class_labels.txt"),
        header=None,
        index=False,
        sep=" ",
    )
    description_df[["img_id", "is_train"]].to_csv(
        os.path.join(root_dir, "train_test_split.txt"),
        header=None,
        index=False,
        sep=" ",
    )


def crop_dogs(root_dir=os.environ.get("DOGS_DIR", "DOGS")):
    """
    Using the `bounding_boxes.txt` file, crop images of dogs dataset.
    """
    # TODO: cropping with bounding boxes may need to be generalized
    # cropping for cub200 is similar to cropping after metadata prep

    root_dir = pathlib.Path(root_dir)

    # read img names, bounding_boxes
    names = pd.read_table(root_dir / "images.txt", delimiter=" ", names=["id", "name"])
    names = names.to_numpy()
    boxs = pd.read_table(
        root_dir / "bounding_boxes.txt",
        delimiter=" ",
        names=["id", "x", "y", "width", "height"],
    )
    boxs = boxs.to_numpy()

    # crop imgs
    imgspath = root_dir / "images_cropped"
    imgspath.mkdir(parents=True, exist_ok=True)
    log.info("Cropping and copying images to %s", imgspath)
    for i, _ in tqdm(enumerate(names)):
        # the RGB conversion avoids RGB-A to JPEG conversion errors
        im = Image.open(root_dir / "Images" / names[i][1]).convert("RGB")
        im = im.crop(
            (boxs[i][1], boxs[i][2], boxs[i][1] + boxs[i][3], boxs[i][2] + boxs[i][4])
        )
        image_cropped_name = imgspath / names[i][1]
        image_cropped_name.parent.mkdir(parents=True, exist_ok=True)
        im.save(image_cropped_name, quality=95)

    log.info("Images copied to %s", imgspath)


class DogsCachedPartLabels(CachedPartLabels):
    def __init__(self, meta_data_path: str, use_parts: bool = False) -> None:
        super().__init__(meta_data_path, use_parts=use_parts)

    def parse_meta_labels(self):
        self.parse_common_meta_labels(cast_id_to_int=False)
        self.parse_part_specific_meta()

    def parse_part_specific_meta(self):
        train_txt = Path(self.meta_data_path, "train_test_split.txt")

        id_to_part_centroid = {}
        with open(train_txt, "r") as f:
            train_lines = f.readlines()
        for train_line in train_lines:
            img_id, _ = train_line.split(" ")[0], int(train_line.split(" ")[1][:-1])
            if img_id not in id_to_part_centroid:
                id_to_part_centroid[img_id] = []

        self.cached_part_id_to_part = None
        self.cached_id_to_part_centroid = id_to_part_centroid
        self.cached_part_num = 0


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("DOGS_DIR", "DOGS"),
    train_dir: str = "train",
    val_dir: str = "validation",
    project_dir: str = None,
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=False,
):
    if part_labels:
        # when part_labels is true, this will fail (dogs does not contain part_labels)
        cached_part_labels = DogsCachedPartLabels(data_path, use_parts=True)
    else:
        cached_part_labels = None

    return torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=120,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir,
        project_dir=project_dir,
    )
