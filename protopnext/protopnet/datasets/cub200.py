import logging
import os
import pathlib
from pathlib import Path
from typing import Union

from ..datasets import torch_extensions
from .torch_extensions import CachedPartLabels

log = logging.getLogger(__name__)


class CUB200CachedPartLabels(CachedPartLabels):
    def parse_meta_labels(self):
        self.parse_common_meta_labels(cast_id_to_int=True)
        self.parse_part_specific_meta()

    def parse_part_specific_meta(self):
        part_cls_txt = Path(self.meta_data_path, "parts", "parts.txt")
        part_loc_txt = Path(self.meta_data_path, "parts", "part_locs.txt")

        # part_id_to_part: Get the part name of each object part according to its part id
        part_id_to_part = {}
        with open(part_cls_txt, "r") as f:
            part_cls_lines = f.readlines()
        for part_cls_line in part_cls_lines:
            id_len = len(part_cls_line.split(" ")[0])
            part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1 :]
            part_id_to_part[int(part_id)] = part_name
        part_num = len(part_id_to_part.keys())

        # id_to_part_loc: Get the part annotations of each image according to its image id
        id_to_part_centroid = {}
        with open(part_loc_txt, "r") as f:
            part_loc_lines = f.readlines()
        for part_loc_line in part_loc_lines:
            content = part_loc_line.split(" ")
            img_id, part_id, loc_x, loc_y, visible = (
                int(content[0]),
                int(content[1]),
                int(float(content[2])),
                int(float(content[3])),
                int(content[4]),
            )
            if img_id not in id_to_part_centroid.keys():
                id_to_part_centroid[img_id] = []
            if visible == 1:
                id_to_part_centroid[img_id].append([part_id, loc_x, loc_y])

        self.cached_part_id_to_part = part_id_to_part
        self.cached_id_to_part_centroid = id_to_part_centroid
        self.cached_part_num = part_num


def train_dataloaders(
    data_path: Union[str, pathlib.Path] = os.environ.get("CUB200_DIR", "CUB_200_2011"),
    train_dir: str = "train",
    val_dir: str = "validation",
    project_dir: str = None,
    image_size=(224, 224),
    batch_sizes={"train": 95, "project": 75, "val": 100},
    part_labels=True,
):
    if part_labels:
        cached_part_labels = CUB200CachedPartLabels(data_path, use_parts=True)
    else:
        cached_part_labels = None

    return torch_extensions.FilesystemSplitDataloaders(
        data_path=data_path,
        num_classes=200,
        image_size=image_size,
        batch_sizes=batch_sizes,
        cached_part_labels=cached_part_labels,
        train_dir=train_dir,
        val_dir=val_dir,
        project_dir=project_dir,
    )
