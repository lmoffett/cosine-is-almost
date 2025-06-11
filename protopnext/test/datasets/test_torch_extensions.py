import copy
import hashlib
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

import protopnet.datasets.torch_extensions as te
from protopnet.datasets.torch_extensions import (
    CombinedDataloader,
    DataclassDataset,
    DictDataLoader,
    DictDataLoaderWithHashedSampleIds,
    FilteredDataset,
    SingleChannelNPDataset,
    SingleSamplePerClassDataloader,
    VisionDictMixin,
    uneven_collate_fn,
)


def test_single_channel_numpy_dataset(temp_dir):
    base_dir = temp_dir / "np-dataset"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir()

    num_classes = 5
    batch_size = 2
    image_size = (224, 224)

    idx = 0
    data_info_dict = {}
    resize = transforms.Resize(image_size)

    for label in range(num_classes):
        class_dir = base_dir / str(label)
        class_dir.mkdir()

        for _ in range(batch_size):
            data = np.random.rand(
                np.random.randint(100, 501), np.random.randint(100, 501)
            )
            file_id = f"tmp_file_{idx}"
            file_name = file_id + ".npy"
            file_path = class_dir / file_name
            np.save(file_path, data)

            data = transforms.Compose(
                [
                    torch.from_numpy,
                ]
            )(data)
            data = resize(data.unsqueeze(0))
            data = data.expand(3, -1, -1)

            data_info_dict[idx] = (torch.linalg.norm(data.float()), file_id, label)

            idx += 1

    customDataSet_kw_args = {
        "root_dir": base_dir,
        "image_size": image_size,
        "fine_annotation": False,
    }

    dataset = SingleChannelNPDataset(**customDataSet_kw_args)

    loader_config = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }

    dataloader = torch.utils.data.DataLoader(dataset, **loader_config)

    num_layers = 3
    desired_batch_dimensions = torch.Size(
        (
            loader_config["batch_size"],
            num_layers,
            customDataSet_kw_args["image_size"][0],
            customDataSet_kw_args["image_size"][1],
        )
    )

    for i, batch in enumerate(dataloader):
        image = batch["img"]
        sample_label = batch["target"]
        sample_id = batch["sample_id"]

        assert image.shape == desired_batch_dimensions
        for j in range(batch_size):
            cur_idx = i * batch_size + j
            expected_norm, id, label = data_info_dict[cur_idx]

            assert sample_id[j] == id, f"Expected {id}, got {sample_id[j]}"
            assert sample_label[j] == label, f"Expected {label}, got {sample_label[j]}"
            norm = torch.linalg.norm(image[j])
            assert norm == expected_norm, f"Expected {expected_norm}, got {norm}"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)


def create_sample_data(num_items, num_centroids, centroid_dim):
    batch = []
    for _ in range(num_items):
        sample = {
            "sample_parts_centroids": torch.randn(num_centroids, centroid_dim),
            "other_tensor": torch.randn((10, 10)),
            "label": torch.randint(0, 5, (1,)).item(),
        }
        batch.append(sample)
    return batch


@pytest.mark.parametrize(
    "num_items, num_centroids, centroid_dim",
    [
        (4, 5, 3),
        (3, 2, 3),
        (5, 1, 3),
    ],
)
def test_uneven_collate_fn(num_items, num_centroids, centroid_dim):
    batch = create_sample_data(num_items, num_centroids, centroid_dim)
    collated = uneven_collate_fn(batch, stack_ignore_key="sample_parts_centroids")

    assert isinstance(collated, dict), "Output should be a dictionary"
    assert isinstance(
        collated["sample_parts_centroids"], list
    ), "Centroids should be in a list"
    assert (
        len(collated["sample_parts_centroids"]) == num_items
    ), "Centroids list should match number of items"
    assert all(
        isinstance(x, torch.Tensor) for x in collated["sample_parts_centroids"]
    ), "Each centroid should be a tensor"
    assert collated["other_tensor"].shape == (
        num_items,
        10,
        10,
    ), "Other tensors should be stacked correctly"
    assert isinstance(
        collated["label"], torch.Tensor
    ), "Labels should be converted to a tensor"
    assert (
        collated["label"].shape[0] == num_items
    ), "Labels tensor should match number of items"


@pytest.mark.parametrize("test_dir", ["test", "fake-test"])
def test_get_dataloaders(temp_dir, test_dir):
    base_dir = temp_dir / "dummy"
    if base_dir.exists():
        shutil.rmtree(base_dir)

    for split in ["train", "test", "validation"]:
        split_dir = base_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        class_dir = split_dir / "dummy_class"
        class_dir.mkdir(parents=True, exist_ok=True)
        # Create a new image with RGB mode
        image = Image.new("RGB", (24, 24), (255, 255, 255))

        # Create a drawing context
        draw = ImageDraw.Draw(image)
        # Draw the circle
        draw.ellipse([6, 6, 18, 18], fill=(0, 0, 255))

        # Save the image
        image.save(class_dir / "BLUE_01.jpg", "JPEG")

    split_dataloaders = te.FilesystemSplitDataloaders(
        base_dir,
        200,
        test_dir=test_dir,
        batch_sizes={"train": 1, "project": 2, "val": 3, "test": 4},
        image_size=(24, 24),
    )

    dataloaders = [
        split_dataloaders.train_loader,
        split_dataloaders.project_loader,
        split_dataloaders.val_loader,
    ]

    if test_dir == "test":
        dataloaders.append(split_dataloaders.test_loader)

    for dl in dataloaders:
        for val_dict in dl:
            assert val_dict["img"].shape[0] > 0
            break

    if test_dir == "fake-test":
        assert hasattr(split_dataloaders, "test_loader") is False


@pytest.fixture(params=["with_fa", "without_fa"])
def mock_np_dataset(temp_dir, request):
    """Returns just the model instance based on the parameterized type."""
    if request.param == "with_fa":
        with_fa = True
    elif request.param == "without_fa":
        with_fa = False

    base_dir = Path(temp_dir / request.param)
    base_dir.mkdir()

    h, w = np.random.randint(100, 501), np.random.randint(100, 501)

    splits = ["train", "push", "validation", "test", "train_auxiliary"]

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        os.mkdir(split_dir)
        class_dir = os.path.join(split_dir, "dummy_class")
        os.mkdir(class_dir)

        for i in range(3):
            if with_fa:
                data = np.random.rand(2, h, w)
            else:
                data = np.random.rand(h, w)
            np.save(os.path.join(class_dir, f"sample{i}.npy"), data)

    yield with_fa, base_dir

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)


@pytest.mark.parametrize("with_aux", [True, False])
@pytest.mark.parametrize("test_dir", ["test", "fake-test"])
def test_get_np_dataloaders(mock_np_dataset, with_aux, test_dir):

    with_fa, data_path = mock_np_dataset

    split_dataloaders = te.NPSplitDataloaders(
        data_path=data_path,
        batch_sizes={
            "train": 2,
            "project": 2,
            "val": 3,
            **({"train_auxiliary": 2} if with_aux else {}),
            "test": 4,
        },
        image_size=24,
        test_dir=test_dir,
        with_fa=with_fa,
        with_aux=with_aux,
    )

    splits = [
        ("train", split_dataloaders.train_loader),
        ("project", split_dataloaders.project_loader),
        ("val", split_dataloaders.val_loader),
    ]

    if test_dir == "test":
        splits.append(("test", split_dataloaders.test_loader))

    for split, dl in splits:
        for val_dict in dl:
            assert val_dict["img"].shape[0] > 0
            if with_fa and (split in ("train", "val", "test")):
                assert val_dict["fine_annotation"].shape[0] == val_dict["img"].shape[0]
            else:
                assert "fine_annotation" not in val_dict

    if test_dir == "fake-test":
        assert hasattr(split_dataloaders, "test_loader") is False


@pytest.fixture
def mock_nested_dataloader():
    # Create a mock dataloader that returns a tuple of (img, target)
    mock_loader = MagicMock()
    mock_loader.batch_size = 2

    img_tensor = torch.rand(2, 3, 32, 32)  # Example 2 images, 3 channels, 32x32 pixels
    target_tensor = torch.tensor([1, 0])  # Example target labels
    sample_id = torch.arange(2)  # Example sample IDs
    mock_loader.__iter__.return_value = [(img_tensor, target_tensor, sample_id)]
    mock_loader.__len__.return_value = 2
    mock_loader.dataset = "mock_dataset"

    return mock_loader


@pytest.fixture
def mock_no_sample_id_nested_dataloader():
    # Create a mock dataloader that returns a tuple of (img, target)
    mock_loader = MagicMock()
    mock_loader.batch_size = 2

    img_tensor = torch.rand(2, 3, 32, 32)  # Example 2 images, 3 channels, 32x32 pixels
    target_tensor = torch.tensor([1, 0])  # Example target labels
    mock_loader.__iter__.return_value = [(img_tensor, target_tensor)]
    mock_loader.__len__.return_value = 2
    mock_loader.dataset = "mock_dataset"

    return mock_loader


@pytest.fixture
def hashing_dict_dataloader(mock_no_sample_id_nested_dataloader):
    return DictDataLoaderWithHashedSampleIds(mock_no_sample_id_nested_dataloader)


@pytest.fixture
def dict_dataloader(mock_nested_dataloader):
    return DictDataLoader(mock_nested_dataloader)


def test_dict_dataloader_iteration(dict_dataloader):
    # iterate over the dataloader twice simultaneously and check if the output is the same
    for batch1, batch2 in zip(dict_dataloader, dict_dataloader):
        for batch in [batch1, batch2]:
            assert "img" in batch
            assert "target" in batch
            assert isinstance(batch["img"], torch.Tensor), type(batch["img"])
            assert isinstance(batch["target"], torch.Tensor), type(batch["target"])
            assert batch["img"].shape == (
                2,
                3,
                32,
                32,
            )
            assert batch["target"].shape == (2,)

        assert torch.equal(batch1["img"], batch2["img"])
        assert torch.equal(batch1["target"], batch2["target"])


def test_hashing_initialization(mock_no_sample_id_nested_dataloader):
    loader = DictDataLoaderWithHashedSampleIds(
        mock_no_sample_id_nested_dataloader, hash_function="md5"
    )
    assert loader.batch_size == mock_no_sample_id_nested_dataloader.batch_size
    assert loader.hash_function == "md5"
    assert loader.dataloader == mock_no_sample_id_nested_dataloader


def test_hashing_iteration(hashing_dict_dataloader):
    # iterate over the dataloader twice simultaneously and check if the output is the same
    for batch1, batch2 in zip(hashing_dict_dataloader, hashing_dict_dataloader):
        for batch in [batch1, batch2]:
            assert "img" in batch
            assert "target" in batch
            assert "sample_id" in batch
            assert isinstance(batch["img"], torch.Tensor), type(batch["img"])
            assert isinstance(batch["target"], torch.Tensor), type(batch["target"])
            assert isinstance(batch["sample_id"], torch.Tensor), type(
                batch["sample_id"]
            )
            assert batch["img"].shape == (
                2,
                3,
                32,
                32,
            )  # Expecting the image shape to match
            assert batch["target"].shape == (
                2,
            )  # Expecting the target to have 2 labels
            assert batch["sample_id"].shape == (
                2,
                32,
            )  # Expecting 2 sample IDs that are 32 bytes long (sha256)

        assert torch.equal(batch1["img"], batch2["img"])
        assert torch.equal(batch1["target"], batch2["target"])
        assert torch.equal(batch1["sample_id"], batch2["sample_id"])


def test_hash_function(hashing_dict_dataloader):
    # Manually hash an image and compare it to the output of the __image_hash_as_sample_ids
    img_tensor = torch.rand(2, 3, 32, 32)
    expected_hash = hashlib.sha256(
        (img_tensor[0].cpu().numpy() * 255).astype("uint8")
    ).digest()

    sample_ids = hashing_dict_dataloader._DictDataLoaderWithHashedSampleIds__image_hash_as_sample_ids(
        img_tensor
    )
    assert isinstance(sample_ids, torch.Tensor)
    assert torch.equal(
        sample_ids[0], torch.frombuffer(expected_hash, dtype=torch.uint8)
    )
    assert sample_ids.shape[0] == 2  # Should generate one hash per image


def test_len(hashing_dict_dataloader, mock_no_sample_id_nested_dataloader):
    assert len(hashing_dict_dataloader) == len(mock_no_sample_id_nested_dataloader)


@dataclass
class TorchishDataclass:
    tensor: torch.Tensor
    label: int
    metadata: str


@pytest.fixture
def sample_dataclass_list():
    # Create a list of MyDataClass instances for testing
    return [
        TorchishDataclass(
            tensor=torch.tensor([1.0, 2.0, 3.0]), label=0, metadata="sample1"
        ),
        TorchishDataclass(
            tensor=torch.tensor([4.0, 5.0, 6.0]), label=1, metadata="sample2"
        ),
        TorchishDataclass(
            tensor=torch.tensor([7.0, 8.0, 9.0]), label=2, metadata="sample3"
        ),
    ]


def test_dataset_length(sample_dataclass_list):
    dataset = DataclassDataset(sample_dataclass_list)
    assert len(dataset) == len(
        sample_dataclass_list
    ), "Dataset length should match the number of data instances."


def test_dataset_getitem(sample_dataclass_list):
    dataset = DataclassDataset(sample_dataclass_list)

    # Test retrieving the first item
    item = dataset[0]
    assert isinstance(item, dict), "Each item should be returned as a dictionary."
    assert (
        "tensor" in item and "label" in item and "metadata" in item
    ), "Each dictionary should contain all fields of the dataclass."
    assert torch.equal(
        item["tensor"], sample_dataclass_list[0].tensor
    ), "Tensor field should match the original data."
    assert (
        item["label"] == sample_dataclass_list[0].label
    ), "Label should match the original data."
    assert (
        item["metadata"] == sample_dataclass_list[0].metadata
    ), "Metadata should match the original data."


def test_dataset_all_items(sample_dataclass_list):
    dataset = DataclassDataset(sample_dataclass_list)

    # Test all items to ensure all fields and values match
    for i in range(len(dataset)):
        item = dataset[i]
        assert torch.equal(
            item["tensor"], sample_dataclass_list[i].tensor
        ), f"Tensor field of item {i} should match."
        assert (
            item["label"] == sample_dataclass_list[i].label
        ), f"Label of item {i} should match."
        assert (
            item["metadata"] == sample_dataclass_list[i].metadata
        ), f"Metadata of item {i} should match."


def test_non_tensor_fields(sample_dataclass_list):
    dataset = DataclassDataset(sample_dataclass_list)
    item = dataset[0]

    # Ensure non-tensor fields are not converted to tensors
    assert not isinstance(
        item["label"], torch.Tensor
    ), "Non-tensor fields should remain as their original types."
    assert not isinstance(
        item["metadata"], torch.Tensor
    ), "Non-tensor fields should remain as their original types."


def test_dataclass_dataset_transforms(sample_dataclass_list):
    def transform_fn(meta):
        meta = copy.copy(meta)
        meta.tensor = meta.tensor * 2
        return meta

    dataset = DataclassDataset(sample_dataclass_list, transform=transform_fn)
    item = dataset[0]

    # Test all items to ensure all fields and values match
    for i in range(len(dataset)):
        item = dataset[i]
        assert torch.equal(
            item["tensor"], sample_dataclass_list[i].tensor * 2
        ), f"Tensor field of item {i} should match."
        assert (
            item["label"] == sample_dataclass_list[i].label
        ), f"Label of item {i} should match."
        assert (
            item["metadata"] == sample_dataclass_list[i].metadata
        ), f"Metadata of item {i} should match."


class MockDataset(Dataset):
    """A mock dataset to simulate the base dataset with 'sample_id' as a key."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def base_dataset():
    # Create a mock dataset with sample_ids from 0 to 4
    data = [{"sample_id": i, "value": i * 10} for i in range(5)]
    return MockDataset(data)


def test_filtered_dataset_length(base_dataset):
    sample_ids = [1, 3]
    filtered_dataset = FilteredDataset(base_dataset, sample_ids)
    assert len(filtered_dataset) == len(
        sample_ids
    ), "Filtered dataset length should match sample_ids length."


def test_filtered_dataset_content(base_dataset):
    sample_ids = [0, 2, 4]
    filtered_dataset = FilteredDataset(base_dataset, sample_ids)

    # Check each sample in the filtered dataset
    for i, sample_id in enumerate(sample_ids):
        sample = filtered_dataset[i]
        assert (
            sample["sample_id"] == sample_id
        ), f"Expected sample_id {sample_id} but got {sample['sample_id']}"
        assert (
            sample["value"] == sample_id * 10
        ), f"Expected value {sample_id * 10} but got {sample['value']}"


def test_filtered_dataset_invalid_sample_id(base_dataset):
    # Sample ID 10 does not exist in the base dataset
    sample_ids = [0, 10]
    with pytest.raises(
        ValueError, match="The following sample_ids are not found in the base dataset"
    ):
        FilteredDataset(base_dataset, sample_ids)


def test_filtered_dataset_empty_sample_ids(base_dataset):
    # Empty sample_ids should raise a ValueError
    sample_ids = []
    with pytest.raises(ValueError, match="sample_ids list cannot be empty"):
        FilteredDataset(base_dataset, sample_ids)


def test_filtered_dataset_all_sample_ids(base_dataset):
    # Include all sample_ids in the dataset
    sample_ids = [0, 1, 2, 3, 4]
    filtered_dataset = FilteredDataset(base_dataset, sample_ids)

    assert len(filtered_dataset) == len(
        sample_ids
    ), "Filtered dataset length should match sample_ids length"
    for i, sample_id in enumerate(sample_ids):
        sample = filtered_dataset[i]
        assert (
            sample["sample_id"] == sample_id
        ), f"Expected sample_id {sample_id} but got {sample['sample_id']}"


class DictMNIST(VisionDictMixin, MNIST):
    pass


def test_vision_mixin(temp_dir):
    mnist = DictMNIST(root=temp_dir / "mnist", download=True)

    assert isinstance(mnist[0], dict)
    assert "img" in mnist[0]
    assert "target" in mnist[0]
    assert "sample_id" in mnist[0] and mnist[0]["sample_id"] == 0


# Expanded parameterized test for testing CombinedDataloader class
@pytest.mark.parametrize(
    "method, expected_len, expected_batch_sizes",
    [
        # 2D Tensor
        pytest.param(
            # temp_dir,
            "continue",
            4,
            [7, 7, 5, 1],
            id="continue method",
        ),
        # 2D Tensor with all zeros
        pytest.param(
            # temp_dir,
            "truncate",
            3,
            [7, 7, 5],
            id="truncate method",
        ),
        # 3D Tensor
        pytest.param(
            # temp_dir,
            "fill",
            4,
            [7, 7, 3 + 2, 1 + 4],
            id="fill method",
        ),
    ],
)
def test_combine_dataloaders(temp_dir, method, expected_len, expected_batch_sizes):
    num_classes = 2
    batch_size1 = 3
    batch_size2 = 4
    image_size = (224, 224)

    idx = 0

    root_dir = temp_dir / "test_combine_dataloaders"
    shutil.rmtree(root_dir, ignore_errors=True)

    for label in range(num_classes):
        class_dir = root_dir / f"{label}"
        os.makedirs(class_dir)
        for _ in range(5):
            data = np.random.rand(2, random.randint(100, 500), random.randint(100, 500))
            file_id = f"temp_file_{idx}"
            file_path = class_dir / f"{file_id}.npy"
            np.save(file_path, data)

            idx += 1

    dataset = SingleChannelNPDataset(
        root_dir=str(root_dir), image_size=image_size, fine_annotation=True
    )

    loader_config1 = {
        "batch_size": batch_size1,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    loader_config2 = {
        "batch_size": batch_size2,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }
    dataloader1 = torch.utils.data.DataLoader(dataset, **loader_config1)
    dataloader2 = torch.utils.data.DataLoader(dataset, **loader_config2)
    dataloader = CombinedDataloader([dataloader1, dataloader2], method=method)

    num_batches = 0
    for i, batch_data_dict in enumerate(dataloader):
        image = batch_data_dict["img"]
        sample_label = batch_data_dict["target"]
        sample_fa = batch_data_dict["fine_annotation"]

        assert (
            image.shape[0] == expected_batch_sizes[i]
        ), f"Image Batch size ({image.shape[0]}) for batch {i} doesn't match expected({expected_batch_sizes[i]})."
        assert (
            sample_label.shape[0] == expected_batch_sizes[i]
        ), f"Label batch size ({sample_label.shape[0]}) for batch {i} doesn't match expected({expected_batch_sizes[i]})."
        assert (
            sample_fa.shape[0] == expected_batch_sizes[i]
        ), f"Label batch size ({sample_fa.shape[0]}) for batch {i} doesn't match expected({expected_batch_sizes[i]})."
        num_batches += 1
    assert (
        num_batches == expected_len
    ), f"Number of batches ({num_batches})doesn't match expected ({expected_len})."


def test_single_sample_per_class_dataloder(temp_dir):
    base_dir = temp_dir / "single-sample-test"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir()

    num_classes = 10
    base_batch_size = 2
    image_size = (224, 224)

    idx = 0
    resize = transforms.Resize(image_size)

    for label in range(num_classes):
        class_dir = base_dir / str(label)
        class_dir.mkdir()

        for _ in range(base_batch_size):
            data = np.random.rand(
                np.random.randint(100, 501), np.random.randint(100, 501)
            )
            file_id = f"tmp_file_{idx}"
            file_name = file_id + ".npy"
            file_path = class_dir / file_name
            np.save(file_path, data)

            data = transforms.Compose(
                [
                    torch.from_numpy,
                ]
            )(data)
            data = resize(data.unsqueeze(0))
            data = data.expand(3, -1, -1)

            idx += 1

    customDataSet_kw_args = {
        "root_dir": base_dir,
        "image_size": image_size,
        "fine_annotation": False,
    }

    dataset = SingleChannelNPDataset(**customDataSet_kw_args)

    loader_config = {
        "batch_size": base_batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": False,
    }

    dataloader = torch.utils.data.DataLoader(dataset, **loader_config)

    batch_size = 5
    single_sample_dataloader = SingleSamplePerClassDataloader(dataloader, batch_size)

    num_layers = 3
    desired_batch_dimensions = torch.Size(
        (
            batch_size,
            num_layers,
            customDataSet_kw_args["image_size"][0],
            customDataSet_kw_args["image_size"][1],
        )
    )

    classes = set()
    for i, batch in enumerate(single_sample_dataloader):
        image = batch["img"]
        sample_label = batch["target"]

        assert image.shape == desired_batch_dimensions

        for label in sample_label:
            if label in classes:
                assert (
                    False
                ), f"Expected unique class labels but got duplicate class {label}"
            classes.add(label)

    assert (
        len(classes) == num_classes
    ), "Did not retrieve sample for every class in dataloader"

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
