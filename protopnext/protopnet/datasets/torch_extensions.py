import hashlib
import logging
import os
import random
import sys
from collections import namedtuple
from dataclasses import InitVar, dataclass, field

# For data configuration
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from ..preprocess import mean, std
from .dataset_prep import calculate_split_mean_std

logger = logging.getLogger(__name__)


class DictDataLoader:
    def __init__(self, dataloader, has_sample_id=False):
        """
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to wrap - should return a tuple of img, target, sample_id
        """
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.has_sample_id = has_sample_id

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        for tup in self.dataloader:
            if self.has_sample_id:
                yield {
                    "img": tup[0],
                    "target": tup[1],
                    "sample_id": tup[2],
                }
            else:
                yield {
                    "img": tup[0],
                    "target": tup[1],
                }

    def __len__(self) -> int:
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def __getattribute__(self, name: str) -> Any:
        if name in [
            "dataloader",
            "batch_size",
            "has_sample_id",
        ]:
            return super().__getattribute__(name)
        elif name in ["dataset"]:
            return self.dataloader.dataset
        return getattr(self.dataloader, name)


class DictDataLoaderWithHashedSampleIds:
    def __init__(self, dataloader, hash_function: str = "sha256"):
        """
        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to wrap - should return a tuple of img, target
            hash_function (str, optional): The hash function to use for hashing the sample IDs
        """
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.hash_function = hash_function

    def __iter__(self) -> Iterable[Dict[str, torch.Tensor]]:
        for img, target in self.dataloader:
            yield {
                "img": img,
                "target": target,
                "sample_id": self.__image_hash_as_sample_ids(img),
            }

    def __len__(self) -> int:
        return len(self.dataloader)

    def __getattribute__(self, name: str) -> Any:
        if name in [
            "dataloader",
            "batch_size",
            "_DictDataLoaderWithHashedSampleIds__image_hash_as_sample_ids",
            "hash_function",
        ]:
            return super().__getattribute__(name)
        elif name in ["dataset"]:
            return self.dataloader.dataset
        return getattr(self.dataloader, name)

    def __image_hash_as_sample_ids(self, images: torch.Tensor) -> str:
        tensor = (images * 255).byte()
        array = tensor.cpu().numpy()

        hashes = []
        for img in array:
            # Generate the hash using the specified hash function
            hasher = hashlib.new(self.hash_function)
            hasher.update(img)
            hashes.append(hasher.digest())

        return torch.tensor(hashes, dtype=torch.uint8)

    @property
    def dataset(self):
        return self.dataloader.dataset


class DataclassDataset(Dataset):
    def __init__(self, dataclass_list: List, transform: callable = None, key_map={}):
        self.data = dataclass_list
        self.transform = transform
        self.key_map = key_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the dataclass instance at the given index
        item = self.data[idx]

        # Create a dictionary for the tensors and other data you want to return
        output = {}
        if self.transform:
            item = self.transform(item)

        # Iterate over all fields in the dataclass
        for this_field in item.__dataclass_fields__.keys():
            value = getattr(item, this_field)
            out_name = self.key_map.get(this_field, this_field)
            output[out_name] = value

        return output


# TODO: replace this with a dataclass
train_loader_set = namedtuple(
    "train_loader_set", ["train_loader", "train_push_loader", "val_loader"]
)


class CachedPartLabels:
    """
    Abstract base class to define the standard interface for dataset metadata handling.
    All datasets inherit common getter methods.
    """

    def __init__(self, meta_data_path: str, use_parts: bool = True) -> None:
        self.meta_data_path = meta_data_path
        self.use_parts = use_parts
        assert Path(
            self.meta_data_path
        ).exists(), f"Metadata path {meta_data_path} does not exist"
        self.cached_id_to_path = {}
        self.cached_path_to_id = {}
        self.cached_id_to_bbox = {}
        self.cached_cls_to_id = {}
        self.cached_id_to_train = {}
        self.cached_part_id_to_part = {}
        self.cached_id_to_part_centroid = {}
        self.cached_part_num = 0

        self.parse_meta_labels()
        self.check_metadata_completeness()

    def check_metadata_completeness(self):
        """Ensures that no essential metadata dictionary is empty."""
        assert self.cached_id_to_path, "id_to_path dictionary is empty"
        assert self.cached_path_to_id, "path_to_id dictionary is empty"
        assert self.cached_id_to_bbox, "id_to_bbox dictionary is empty"
        assert self.cached_cls_to_id, "cls_to_id dictionary is empty"
        assert self.cached_id_to_train, "id_to_train dictionary is empty"
        assert (
            self.cached_part_id_to_part or not self.use_parts
        ), "part_id_to_part dictionary is empty"
        assert (
            self.cached_id_to_part_centroid or not self.use_parts
        ), "id_to_part_centroid dictionary is empty"
        assert (
            self.cached_part_num > 0 or not self.use_parts
        ), "No parts are defined in part_num"

    def parse_common_meta_labels(self, cast_id_to_int=True):
        img_txt = Path(self.meta_data_path, "images.txt")
        cls_txt = Path(self.meta_data_path, "image_class_labels.txt")
        bbox_txt = Path(self.meta_data_path, "bounding_boxes.txt")
        train_txt = Path(self.meta_data_path, "train_test_split.txt")

        # id_to_path: Get the image path of each image according to its image id
        cached_id_to_path = {}
        with open(img_txt, "r") as f:
            img_lines = f.readlines()
        for img_line in img_lines:
            if cast_id_to_int:
                img_id, img_path = (
                    int(img_line.split(" ")[0]),
                    img_line.split(" ")[1][:-1],
                )
            else:
                img_id, img_path = img_line.split(" ")[0], img_line.split(" ")[1][:-1]
            img_folder, img_name = img_path.split("/")[0], img_path.split("/")[1]
            cached_id_to_path[img_id] = (img_folder, img_name)

        # id_to_bbox: Get the bounding box annotation (bird part) of each image according to its image id
        cached_id_to_bbox = {}
        with open(bbox_txt, "r") as f:
            bbox_lines = f.readlines()
        for bbox_line in bbox_lines:
            cts = bbox_line.split(" ")
            img_id, bbox_x, bbox_y, bbox_width, bbox_height = (
                int(cts[0]) if cast_id_to_int else cts[0],
                int(cts[1].split(".")[0]),
                int(cts[2].split(".")[0]),
                int(cts[3].split(".")[0]),
                int(cts[4].split(".")[0]),
            )
            bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
            cached_id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

        # cls_to_id: Get the image ids of each class
        cls_to_id = {}
        with open(cls_txt, "r") as f:
            cls_lines = f.readlines()
        for cls_line in cls_lines:
            img_id, cls_id = (
                (
                    int(cls_line.split(" ")[0])
                    if cast_id_to_int
                    else cls_line.split(" ")[0]
                ),
                int(cls_line.split(" ")[1]) - 1,
            )  # 0 -> 199
            if cls_id not in cls_to_id.keys():
                cls_to_id[cls_id] = []
            cls_to_id[cls_id].append(img_id)

        # id_to_train: Get the training/test label of each image according to its image id
        id_to_train = {}
        with open(train_txt, "r") as f:
            train_lines = f.readlines()
        for train_line in train_lines:
            if cast_id_to_int:
                img_id, is_train = int(train_line.split(" ")[0]), int(
                    train_line.split(" ")[1][:-1]
                )
            else:
                img_id, is_train = train_line.split(" ")[0], int(
                    train_line.split(" ")[1][:-1]
                )
            id_to_train[img_id] = is_train

        path_to_id = {"_".join(v): k for k, v in cached_id_to_path.items()}

        self.cached_id_to_path = cached_id_to_path
        self.cached_path_to_id = path_to_id
        self.cached_id_to_bbox = cached_id_to_bbox
        self.cached_cls_to_id = cls_to_id
        self.cached_id_to_train = id_to_train

    def parse_meta_labels(self):
        """
        Parses the dataset-specific metadata files. Needs to be implemented.
        """
        pass

    def id_to_path(self, id: int) -> Tuple[str, str]:
        """Return the path corresponding to a given ID."""
        return self.cached_id_to_path.get(id)

    def path_to_id(self, path: str) -> int:
        """Return the ID corresponding to a given path."""
        return self.cached_path_to_id.get(path)

    def id_to_bbox(self, id: int) -> Tuple[int, int, int, int]:
        """Return the bounding box corresponding to a given ID."""
        return self.cached_id_to_bbox.get(id)

    def cls_to_id(self, cls_id: int) -> List[int]:
        """Return the IDs corresponding to a given class."""
        return self.cached_cls_to_id.get(cls_id, [])

    def id_to_train(self, id: int) -> bool:
        """Return the binary training/test flag corresponding to a given ID."""
        return self.cached_id_to_train.get(id)

    def part_id_to_part(self, part_id: int) -> str:
        """Return the part name corresponding to a given part ID."""
        return self.cached_part_id_to_part.get(part_id)

    def id_to_part_centroid(self, id: int) -> List[Tuple[int, int, int]]:
        """Return the part locations corresponding to a given ID."""
        return self.cached_id_to_part_centroid.get(id, [])

    def get_part_num(self) -> int:
        """Return the number of parts managed by the dataset."""
        return self.cached_part_num


@dataclass
class FilesystemSplitDataloaders:
    data_path: InitVar[Union[str, Path]]
    num_classes: int
    batch_sizes: InitVar[Dict[str, int]]
    image_size: Tuple[int, int]
    cached_part_labels: InitVar[Union[CachedPartLabels, None]] = None
    train_dir: str = "train"
    val_dir: str = "validation"
    project_dir: Optional[str] = None
    test_dir: str = "test"
    train_loader: torch.utils.data.DataLoader = field(init=False)
    project_loader: torch.utils.data.DataLoader = field(init=False)
    val_loader: torch.utils.data.DataLoader = field(init=False)
    test_loader: Optional[torch.utils.data.DataLoader] = field(init=False)
    augment: bool = False

    def __post_init__(self, data_path, batch_sizes, cached_part_labels):
        """
        Create train, push and validation dataloaders for the specified dataset.
        """
        data_path = Path(data_path)
        train_dir = data_path / self.train_dir
        val_dir = data_path / self.val_dir
        project_dir = data_path / self.project_dir if self.project_dir else train_dir
        test_dir = data_path / self.test_dir

        seed = int(os.environ.get("PPNXT_SEED", 1234))
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_distribution = calculate_split_mean_std(
            ImageFolderDict(
                train_dir,
                transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Resize(
                            size=(self.image_size[0], self.image_size[1])
                        ),
                    ]
                ),
            )
        )
        normalize = transforms.Normalize(
            mean=train_distribution.mean, std=train_distribution.std
        )

        if self.augment:  # added for ProtoTree
            transform = transforms.Compose(
                [
                    transforms.Resize(size=(self.image_size[0], self.image_size[1])),
                    transforms.RandomOrder(
                        [
                            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                            transforms.ColorJitter(
                                (0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (-0.02, 0.02)
                            ),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(
                                degrees=10, shear=(-2, 2), translate=[0.05, 0.05]
                            ),
                        ]
                    ),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
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
                    transforms.Resize(size=(self.image_size[0], self.image_size[1])),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        train_dataset = ImageFolderDict(
            train_dir,
            transform,
        )

        train_loader_config = {
            "batch_size": batch_sizes["train"],
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
        }

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, **train_loader_config
        )

        project_dataset = ImageFolderDict(
            project_dir,
            transforms.Compose(
                [
                    transforms.Resize(size=(self.image_size[0], self.image_size[1])),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        project_dataset_config = {
            "batch_size": batch_sizes["project"],
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
        }

        self.project_loader = torch.utils.data.DataLoader(
            project_dataset, **project_dataset_config
        )

        val_dataset = ImageFolderDict(
            val_dir,
            transforms.Compose(
                [
                    transforms.Resize(size=(self.image_size[0], self.image_size[1])),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            cached_part_labels=cached_part_labels,
        )

        val_loader_config = {
            "batch_size": batch_sizes["val"],
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "collate_fn": partial(
                uneven_collate_fn, stack_ignore_key="sample_parts_centroids"
            ),
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
        }

        self.val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

        if test_dir.exists():
            test_dataset = ImageFolderDict(
                test_dir,
                transforms.Compose(
                    [
                        transforms.Resize(
                            size=(self.image_size[0], self.image_size[1])
                        ),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
                cached_part_labels=cached_part_labels,
            )

            test_loader_config = val_loader_config.copy()
            if "test" in batch_sizes:
                test_loader_config["batch_size"] = batch_sizes["test"]
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, **test_loader_config
            )
        else:
            logger.warning(
                f"Test directory {test_dir} does not exist. Skipping test set."
            )


@dataclass
class NPSplitDataloaders:
    data_path: InitVar[Union[str, Path]]
    batch_sizes: InitVar[Dict[str, int]]
    image_size: Tuple[int, int]
    train_dir: str = "train"
    project_dir: str = "push"
    val_dir: str = "validation"
    test_dir: str = "test"
    train_aux_dir: str = "train_auxiliary"
    train_loader: torch.utils.data.DataLoader = field(init=False)
    project_loader: torch.utils.data.DataLoader = field(init=False)
    val_loader: torch.utils.data.DataLoader = field(init=False)
    test_loader: Optional[torch.utils.data.DataLoader] = field(init=False)
    with_fa: bool = True
    with_aux: bool = True
    combine_datasets_method: str = "continue"

    def __post_init__(self, data_path, batch_sizes):
        """
        Create train, push and validation dataloaders for the specified dataset.
        """
        train_dir = os.path.join(data_path, self.train_dir)
        project_dir = os.path.join(data_path, self.project_dir)
        val_dir = os.path.join(data_path, self.val_dir)
        train_aux_dir = os.path.join(data_path, self.train_aux_dir)
        test_dir = os.path.join(data_path, self.test_dir)

        seed = int(os.environ.get("PPNXT_SEED", 1234))
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_distribution = calculate_split_mean_std(
            SingleChannelNPDataset(
                root_dir=train_dir,
                image_size=self.image_size,
            ),
        )

        normalize = transforms.Normalize(
            mean=train_distribution.mean, std=train_distribution.std
        )

        train_dataset = SingleChannelNPDataset(
            root_dir=train_dir,
            image_size=self.image_size,
            fine_annotation=self.with_fa,
            transform=normalize,
        )

        self.num_classes = train_dataset.num_classes

        train_loader_config = {
            "batch_size": batch_sizes["train"],
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
            "collate_fn": sparse_collate_fn,
        }

        train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)

        if self.with_aux:
            train_aux_dataset = SingleChannelNPDataset(
                root_dir=train_aux_dir,
                image_size=self.image_size,
                fine_annotation=self.with_fa,
            )

            train_aux_loader_config = {
                "batch_size": batch_sizes["train_auxiliary"],
                "shuffle": True,
                "num_workers": 2,
                "pin_memory": False,
                "prefetch_factor": 8,
                "collate_fn": sparse_collate_fn,
            }

            train_aux_loader = torch.utils.data.DataLoader(
                train_aux_dataset, **train_aux_loader_config
            )

            sparse_keys = []
            if self.with_fa:
                sparse_keys.append("fine_annotation")

            self.train_loader = CombinedDataloader(
                [train_loader, train_aux_loader],
                method=self.combine_datasets_method,
                sparse_keys=sparse_keys,
            )
        else:
            self.train_loader = train_loader

        project_dataset = SingleChannelNPDataset(
            root_dir=project_dir,
            image_size=self.image_size,
            fine_annotation=False,
            transform=normalize,
        )

        project_loader_config = {
            "batch_size": batch_sizes["project"],
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
        }

        self.project_loader = torch.utils.data.DataLoader(
            project_dataset, **project_loader_config
        )

        val_dataset = SingleChannelNPDataset(
            root_dir=val_dir,
            image_size=self.image_size,
            fine_annotation=self.with_fa,
            transform=normalize,
        )

        val_loader_config = {
            "batch_size": batch_sizes["val"],
            "shuffle": False,
            "num_workers": 2,
            "pin_memory": False,
            "prefetch_factor": 8,
            "generator": generator,
            "worker_init_fn": seed_worker_for_reproducability,
            "collate_fn": sparse_collate_fn,
        }

        self.val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

        if Path(test_dir).exists():
            test_dataset = SingleChannelNPDataset(
                root_dir=test_dir,
                image_size=self.image_size,
                fine_annotation=self.with_fa,
                transform=normalize,
            )

            test_loader_config = val_loader_config.copy()
            if "test" in batch_sizes:
                test_loader_config["batch_size"] = batch_sizes["test"]
            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, **test_loader_config
            )


def train_dataloaders(
    # TODO should be pathlib.Path
    data_path,
    cached_part_labels: CachedPartLabels,
    train_dir="train",
    val_dir="validation",
    batch_sizes={"train": 95, "push": 75, "val": 100},
    seed=int(os.environ.get("PPNXT_SEED", 1234)),
) -> train_loader_set:
    """
    Create train, push and validation dataloaders for the specified dataset.
    """
    train_dir = os.path.join(data_path, train_dir)
    val_dir = os.path.join(data_path, val_dir)

    generator = torch.Generator()
    generator.manual_seed(seed)

    img_size = 224
    normalize = transforms.Normalize(mean=mean, std=std)

    train_dataset = ImageFolderDict(
        train_dir,
        transforms.Compose(
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
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_loader_config = {
        "batch_size": batch_sizes["train"],
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 8,
        "generator": generator,
        "worker_init_fn": seed_worker_for_reproducability,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_config)

    train_push_dataset = ImageFolderDict(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    push_loader_config = {
        "batch_size": batch_sizes["push"],
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 8,
        "generator": generator,
        "worker_init_fn": seed_worker_for_reproducability,
    }

    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, **push_loader_config
    )

    val_dataset = ImageFolderDict(
        val_dir,
        transforms.Compose(
            [
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        cached_part_labels=cached_part_labels,
    )

    val_loader_config = {
        "batch_size": batch_sizes["val"],
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "prefetch_factor": 8,
        "collate_fn": partial(
            uneven_collate_fn, stack_ignore_key="sample_parts_centroids"
        ),
        "generator": generator,
        "worker_init_fn": seed_worker_for_reproducability,
    }

    val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_config)

    return train_loader_set(train_loader, train_push_loader, val_loader)


class RandomDataset(data.Dataset):
    def __init__(self, data_size, mode):
        self.data_size = data_size
        self.mode = mode

    def __getitem__(self, index):
        # Set the random seed for generating random numbers
        torch.manual_seed(42)

        # Generate random tensor data
        random_data = torch.rand(
            ((3, 167, 20))
        )  # Example: random tensor of size [3, 32, 32]

        # Generate random tensor label
        random_label = torch.randint(0, 3, (1,))[0]

        return {
            "img": random_data,
            "target": random_label,
            "sample_id": index,
        }  # Return the random data and index in a dictionary

    def __len__(self):
        return self.data_size


class SingleChannelNPDataset(Dataset):
    def __init__(
        self,
        *,
        root_dir,
        image_size,
        transform=None,
        fine_annotation=False,
    ):
        super(SingleChannelNPDataset).__init__()

        self.root_dir = root_dir
        self.image_size = image_size

        self.image_resize = transforms.Resize(size=self.image_size)

        self.fine_annotation = fine_annotation

        self.transform = transform

        classes, class_to_idx = self._find_classes(self.root_dir)
        self.num_classes = len(classes)
        self.samples = self._make_dataset(
            root_dir=self.root_dir, class_to_idx=class_to_idx
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample_id = path.split("/")[-1].split(".npy")[0]
        sample = np.load(path)
        sample = transforms.Compose(
            [
                torch.from_numpy,
            ]
        )(sample)

        if not self.fine_annotation:
            if len(sample.shape) == 3:
                sample = sample[0]

            fa_term = {}
        else:
            if len(sample.shape) == 3:
                fine_annotation = self.image_resize(sample[1].unsqueeze(0))
                sample = sample[0]
            else:
                fine_annotation = torch.sparse_coo_tensor(size=(1, *self.image_size))

            fa_term = {"fine_annotation": fine_annotation}

        sample = self.image_resize(sample.unsqueeze(0))

        if sample.shape[0] == 1:
            sample = sample.expand(3, -1, -1)

        sample_dict = {
            "img": sample.float(),
            "target": target,
            "sample_id": sample_id,
            **fa_term,
        }

        return sample_dict

    def _find_classes(self, root_dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        else:
            classes = [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, root_dir, class_to_idx):
        images = []

        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
        return images


class ImageFolderDict(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=default_loader,
        is_valid_file=None,
        cached_part_labels: CachedPartLabels = None,
    ):
        super(ImageFolderDict, self).__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        self.cached_part_labels = cached_part_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: A dict containing a sample with key "img", the corresponding label with key "target".
                and the sample ID with key "sample_id".
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        ori_sample_wh = sample.size

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample_dict = {"img": sample, "target": target, "sample_id": str(index)}
        if self.cached_part_labels is not None:
            original_id = self.cached_part_labels.path_to_id(
                "_".join(path.split("/")[-2:])
            )
            assert (
                original_id is not None
            ), f"Original ID {original_id} from {path} not found in metadata file"

            # convert all part labels to 0-1 scale
            bbox = torch.tensor(self.cached_part_labels.id_to_bbox(original_id)).float()
            bbox[0] /= ori_sample_wh[0]
            bbox[1] /= ori_sample_wh[1]
            bbox[2] /= ori_sample_wh[0]
            bbox[3] /= ori_sample_wh[1]
            bbox = torch.clamp(bbox, max=1.0)

            if self.cached_part_labels.use_parts:
                part_centroid = torch.tensor(
                    self.cached_part_labels.id_to_part_centroid(original_id)
                ).float()
                part_centroid[:, 1] /= ori_sample_wh[0]
                part_centroid[:, 2] /= ori_sample_wh[1]
                part_centroid[:, 1:] = torch.clamp(part_centroid[:, 1:], max=1.0)

                sample_dict["sample_parts_centroids"] = part_centroid
            else:
                part_centroid = torch.tensor(
                    self.cached_part_labels.id_to_part_centroid(original_id)
                ).float()
                sample_dict["sample_parts_centroids"] = part_centroid

            sample_dict["sample_bounding_box"] = bbox

        return sample_dict


class TensorToDictDatasetAdapter(data.Dataset):
    """
    Simple adapter for a tensor dataset that relies on ordering of the returns to create
    a dictionary dataset compatible with protopnext dataset format.
    """

    def __init__(self, tensor_dataset):
        """
        Args:
            tensor_dataset (torch.utils.data.Dataset): The tensor dataset to adapt
        """
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        """
        Returns a dictionary with the sample data and target
        """
        sample = self.tensor_dataset[index]
        if hasattr(sample, "__iter__"):
            if len(sample) == 2:
                return {"img": sample[0], "target": sample[1]}
            elif len(sample) == 3:
                return {"img": sample[0], "target": sample[1], "sample_id": sample[2]}
            else:
                raise NotImplementedError("Expected sample to be length 1, 2, or 3")
        else:
            return {"img": sample}


class TensorDatasetDict(torch.utils.data.TensorDataset):
    """
    A simple extension of the PyTorch TensorDataset that returns a dictionary
    with the sample data and target.
    """

    def __init__(self, *args, **kwargs):
        super(TensorDatasetDict, self).__init__(*args, **kwargs)

    def __len__(self):
        return super(TensorDatasetDict, self).__len__()

    def __getitem__(self, index):
        img, target = super(TensorDatasetDict, self).__getitem__(index)
        return {"img": img, "target": target, "sample_id": index}


class FilteredDataset(Dataset):
    def __init__(self, base_dataset, sample_ids, transform=None):
        """
        Initialize the filtered dataset.

        Args:
            base_dataset (Dataset): The original dataset to wrap.
            sample_ids (list): List of sample IDs to filter by. The length of this list will
                                determine the length of the filtered dataset.
        """
        self.base_dataset = base_dataset
        self.sample_ids = sample_ids
        self.transform = transform

        # Check if length of sample_ids matches the filtered length
        filtered_length = len(self.sample_ids)
        if filtered_length == 0:
            raise ValueError("sample_ids list cannot be empty.")

        # Create a mapping of sample_id to index in the base dataset
        self.id_to_index = {
            self.base_dataset[i]["sample_id"]: i for i in range(len(self.base_dataset))
        }

        # Check that all sample_ids exist in the base dataset
        missing_ids = [sid for sid in self.sample_ids if sid not in self.id_to_index]
        if missing_ids:
            raise ValueError(
                f"The following sample_ids are not found in the base dataset: {missing_ids}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        """
        Retrieve the item corresponding to the filtered sample ID.

        Args:
            idx (int): Index in the filtered dataset.

        Returns:
            dict: Data sample from the base dataset corresponding to sample_ids[idx].
        """
        # Get the index in the base dataset
        base_idx = self.id_to_index[self.sample_ids[idx]]
        sample = self.base_dataset[base_idx]

        if self.transform:
            sample["img"] = self.transform(sample)

        return sample


class VisionDictMixin:
    def __getitem__(self, index: int) -> Dict:
        item = super().__getitem__(index)

        return {
            "img": item[0],
            "target": item[1],
            "sample_id": index,
        }


def uneven_collate_fn(batch, stack_ignore_key):
    """
    Collates a batch of data similar to default stacking collate fn. However,
    this function zips the key entries that have uneven dimensions (number of
    elements). This is useful when the data samples have different number of values.
    For example, image samples may have different number of visible, labeled parts.

    Parameters:
    - batch (list of dicts): A batch of data where each item is a dictionary
        representing one data sample.
    - stack_ignore_key (str, optional): The key for which custom collation is to be
        bypassed. This key's data will not be stacked into a tensor.

    Returns:
    - dict: A dictionary where keys correspond to the keys in the original data
        samples, and values are the data from each sample collated. All values in
        this dict will be tensors except for the values for 'stack_ignore_key', which
        will be a list of tensors. The list will contain a tensor for each sample,
        with the first dimension of each tensor representing the index of whatever
        is being stacked.
    """

    batched_data = {}

    for key in batch[0].keys():
        batched_data[key] = []
    for item in batch:
        for key in item:
            batched_data[key].append(item[key])

    for key in batched_data:
        if all(isinstance(x, torch.Tensor) for x in batched_data[key]):
            if key != stack_ignore_key:  # We already handled 'sample_parts_centroids'
                batched_data[key] = torch.stack(batched_data[key])
        if all(isinstance(x, int) for x in batched_data[key]):
            batched_data[key] = torch.tensor(batched_data[key])

    return batched_data


def sparse_collate_fn(batch):
    batched_data = {}

    for key in batch[0].keys():
        batched_data[key] = []
    for item in batch:
        for key in item:
            if isinstance(item[key], torch.Tensor) and item[key].is_sparse:
                batched_data[key].append(item[key].to_dense())
            else:
                batched_data[key].append(item[key])

    for key in batched_data:
        if all(isinstance(x, torch.Tensor) for x in batched_data[key]):
            batched_data[key] = torch.stack(batched_data[key])
        if all(isinstance(x, int) for x in batched_data[key]):
            batched_data[key] = torch.tensor(batched_data[key])

    return batched_data


class CombinedDataloader(DataLoader):
    """
    Simple adapter to combine multiple dataloaders with different batch sizes. Note: will only use the return dict keys of the first dataloader.
    """

    def __init__(
        self,
        dataloaders: List[DataLoader],
        method: Optional[str] = "continue",
        sparse_keys: List[str] = None,
    ):
        """
        Args:
            dataloaders (List[torch.utils.data.DataLoader]): A list of dataloders to combine
            method (Optional[str]): the method used for combining dataloaders of different lengths.
                'fill' will make sure to go through all the data and each batch has data from all dataloaders, even if it requries repeating data.
                'truncate' will stop iterating once we reach the end of any of the dataloaders.
                'continue' will use all the data but if one or more of the dataloaders has reached its end, it will not repeat data.
        """
        assert method in [
            "fill",
            "truncate",
            "continue",
        ], "The 'method' argument must be either 'fill', 'truncate', or 'continue'"
        self.method = method
        self.dataloaders = dataloaders
        self.batch_size = sum([d.batch_size for d in dataloaders])
        self.lens = [len(d) for d in dataloaders]
        self.sparse_keys = sparse_keys

    def __iter__(self):
        _dataloaders = [iter(d) for d in self.dataloaders]

        for _ in range(self.__len__()):
            out = {}
            for i, dataloader in enumerate(_dataloaders):
                try:
                    data = next(dataloader)
                except StopIteration:
                    if self.method == "fill":
                        _dataloaders[i] = iter(self.dataloaders[i])
                        data = next(_dataloaders[i])
                    elif self.method == "truncate":
                        raise StopIteration
                    elif self.method == "continue":
                        continue

                if not out:
                    out = data
                else:
                    for k in data.keys():
                        if isinstance(data[k], torch.Tensor):
                            if data[k].is_sparse:
                                data[k] = data[k].to_dense()
                            out[k] = torch.cat((out[k], data[k]), 0)
                        elif isinstance(data[k], list):
                            out[k] += data[k]

            yield out

    def __len__(self):
        """
        Returns the number of batches.
        """
        if self.method == "truncate":
            return min(self.lens)
        else:
            return max(self.lens)


class SingleSamplePerClassDataloader(DataLoader):
    def __init__(self, dataloader: DataLoader, batch_size: int = None):
        self.dataloader = dataloader
        self.class_samples = None

        self._get_class_samples()

        if batch_size is None:
            self.batch_size = self.class_samples["target"].shape[0]
        else:
            self.batch_size = batch_size

    def __iter__(self):
        out = {}
        n_samples = self.class_samples["target"].shape[0]

        for idx in range(n_samples):
            if not out:
                for k in self.class_samples.keys():
                    if isinstance(self.class_samples[k][idx], torch.Tensor):
                        out[k] = self.class_samples[k][idx].unsqueeze(0)
                    elif isinstance(self.class_samples[k][idx], list):
                        out[k] = self.class_samples[k][idx]
            else:
                for k in self.class_samples.keys():
                    if isinstance(self.class_samples[k][idx], torch.Tensor):
                        out[k] = torch.cat(
                            (out[k], self.class_samples[k][idx].unsqueeze(0)), 0
                        )
                    elif isinstance(self.class_samples[k][idx], list):
                        out[k] += self.class_samples[k][idx]

            if out["target"].shape[0] == self.batch_size:
                yield out
                out = {}

        if out:
            yield out

    def __len__(self) -> int:
        return self.batch_size

    def _get_class_samples(self):
        out = {}

        for batch in self.dataloader:
            target = batch["target"]

            for idx, target_ in enumerate(target):
                if not out:
                    for k in batch.keys():
                        if isinstance(batch[k][idx], torch.Tensor):
                            out[k] = batch[k][idx].unsqueeze(0)
                        elif isinstance(batch[k][idx], list):
                            out[k] = batch[k][idx]
                else:
                    if target_ not in out["target"]:
                        for k in batch.keys():
                            if isinstance(batch[k][idx], torch.Tensor):
                                out[k] = torch.cat(
                                    (out[k], batch[k][idx].unsqueeze(0)), 0
                                )
                            elif isinstance(batch[k][idx], list):
                                out[k] += batch[k][idx]

        self.class_samples = out


def seed_worker_for_reproducability(worker_id):
    """
    Hook for worker initialization to set the seed for each worker.

    See: https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = int(os.environ.get("PPNXT_SEED", 1234)) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
