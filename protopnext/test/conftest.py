import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader


@pytest.fixture(scope="package")
def temp_root_dir():
    # Check if the environment variable PROTOPNET_TEST_TMP is set
    test_tmp = os.environ.get("PROTOPNET_TEST_TMP")

    if test_tmp:
        # Yield the directory from the environment variable as a pathlib.Path
        # This WILL NOT automatically be cleaned up
        yield Path(test_tmp)
    else:
        # Use tempfile to create a temporary directory and yield it
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)  # Yield the temp directory as a pathlib.Path


@pytest.fixture(scope="module")
def temp_dir(request, temp_root_dir):
    """
    Fixture that provides a test-specific temporary directory path.

    The path is structured as:
    ${temp_root_dir}/${test_dir}/${test_file_name}
    """
    # Get the test file's full path
    test_file_path = request.fspath
    # Split into directory and file name
    test_dir, test_file_name = os.path.split(test_file_path)
    # If the test file is in a directory, include the relative path in temp_dir
    relative_test_dir = os.path.relpath(test_dir, start=os.getcwd())
    # Construct the specific directory path
    if relative_test_dir == ".":
        # If the file is in the current directory
        test_specific_path = os.path.join(temp_root_dir, test_file_name)
    else:
        # Include the relative test directory
        test_specific_path = os.path.join(
            temp_root_dir, relative_test_dir, test_file_name
        )
    # Ensure the directory exists
    os.makedirs(test_specific_path, exist_ok=True)
    return Path(test_specific_path)


@pytest.fixture(scope="function", autouse=True)
def seed():
    """
    Resets the seeds for every test.
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    return seed


class ShortCIFAR10DictDataset(torchvision.datasets.CIFAR10):
    """
    Small CIFAR10 dataset for testing.
    """

    def __init__(self, use_ind_as_label=False, *args, **kwargs):
        """
        use_index_as_label: bool - use the index as the label for the image.
        """
        super(ShortCIFAR10DictDataset, self).__init__(*args, **kwargs)
        self.use_ind_as_label = use_ind_as_label

    def __len__(self):
        return 10

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dict: {'img': image, 'target': target, 'sample_id': sample_id} where target is index of the target class.
        """

        img = self.data[index]
        if self.use_ind_as_label:
            # This is useful for class specific push, where we
            # want to make sure we have at least one image from
            # each class.
            target = torch.tensor(index)
        else:
            target = self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": img, "target": target, "sample_id": index}


@pytest.fixture
def short_cifar10(temp_dir):
    return ShortCIFAR10DictDataset(
        root=temp_dir / "short_cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )


@pytest.fixture
def short_cifar10_one_class_per_sample(temp_dir):
    return ShortCIFAR10DictDataset(
        use_ind_as_label=True,
        root=temp_dir / "short_cifar10",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )


@pytest.fixture
def mock_protopnet():
    class MockProtoPNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(10, 10)
            self.add_on_layers = torch.nn.Linear(10, 10)

            # Create a simple module for prototype layer
            class PrototypeLayer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.prototype_tensors = torch.nn.Parameter(torch.randn(4, 3))

            self.prototype_layer = PrototypeLayer()

            # Create prediction head
            class PredictionHead(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.class_connection_layer = torch.nn.Linear(10, 2)

            self.prototype_prediction_head = PredictionHead()

            # Add required methods that aren't part of nn.Module
            self.project = lambda x: None
            self.prototypes_embedded = lambda: True

        def forward(
            self,
            x: torch.Tensor,
            return_prototype_layer_output_dict: bool = False,
            **kwargs,
        ):
            return {
                "similarity_score_to_each_prototype": torch.randn(len(x), 4, 3, 5),
                "logits": torch.randn(len(x), 2),
            }

    return MockProtoPNet()


@pytest.fixture
def mock_prototree():
    class MockProtoTree(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(10, 10)
            self.add_on_layers = torch.nn.Linear(10, 10)

            # Create a simple module for prototype layer
            class PrototypeLayer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.prototype_tensors = torch.nn.Parameter(torch.randn(4, 3))

            self.prototype_layer = PrototypeLayer()

            # Create mock leaf class
            class MockLeaf:
                def __init__(self, num_classes=2):
                    self._dist_params = torch.nn.Parameter(torch.randn(num_classes))

            # Create prediction head with prototree structure
            class PredictionHead(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.class_connection_layer = torch.nn.Linear(10, 2)

                    # Create a mock tree structure with leaves
                    class MockProtoTree:
                        def __init__(self):
                            self.leaves = [
                                MockLeaf(),
                                MockLeaf(),
                                MockLeaf(),
                            ]  # Create some mock leaves

                    self.prototree = MockProtoTree()

            self.prototype_prediction_head = PredictionHead()

            # Add required methods that aren't part of nn.Module
            self.project = lambda x: None
            self.prototypes_embedded = lambda: True

            self._old_leaf_dist = None  # For tracking leaf distributions

        def forward(
            self,
            x: torch.Tensor,
            return_prototype_layer_output_dict: bool = False,
            **kwargs,
        ):
            return {
                "similarity_score_to_each_prototype": torch.randn(len(x), 4, 3, 5),
                "logits": torch.randn(len(x), 2),
            }

        def batch_derivative_free_update(
            self, output, target, num_batches, old_leaf_dist
        ):
            self._old_leaf_dist = old_leaf_dist  # Store for testing purposes
            # Mock implementation that can be tracked in tests
            pass

        def reinit_parameters(self):
            # Mock implementation of parameter reinitialization
            with torch.no_grad():
                if hasattr(self.prototype_layer, "prototype_tensors"):
                    torch.nn.init.normal_(
                        self.prototype_layer.prototype_tensors, 0.5, 0.1
                    )

        def prune_prototypes(self):
            # Mock implementation returning some prototype indices to prune
            return [1, 2]  # Return dummy indices for testing

    # Create and return the mock instance
    return MockProtoTree()


@pytest.fixture
def mock_deformable():
    class MockDeformable(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(10, 10)
            self.add_on_layers = torch.nn.Linear(10, 10)

            # Create a simple module for prototype layer with offset predictor
            class PrototypeLayer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.prototype_tensors = torch.nn.Parameter(torch.randn(1, 10))

                    class OffsetPredictor(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.offset = torch.nn.Parameter(torch.randn(1, 10))

                    self.offset_predictor = OffsetPredictor()

            self.prototype_layer = PrototypeLayer()

            # Create prediction head
            class PredictionHead(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.class_connection_layer = torch.nn.Linear(10, 2)

            self.prototype_prediction_head = PredictionHead()

            # Add required methods that aren't part of nn.Module
            self.project = lambda x: None
            self.prototypes_embedded = lambda: True

        def forward(
            self,
            x: torch.Tensor,
            return_prototype_layer_output_dict: bool = False,
            **kwargs,
        ):
            return {
                "similarity_score_to_each_prototype": torch.randn(len(x), 3, 4, 5),
                "logits": torch.randn(len(x), 10),
            }

    return MockDeformable()


@pytest.fixture
def mock_stprotopnet(mock_protopnet):
    class MockSTProtoPNet(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Add required methods that aren't part of nn.Module
            self.project = lambda x: None
            self.prototypes_embedded = lambda: True
            self.models = [mock_protopnet, copy.deepcopy(mock_protopnet)]

            self.backbone = self.models[0].backbone
            self.add_on_layers = self.models[0].add_on_layers

        def forward(
            self,
            x: torch.Tensor,
            additional_forward_requirements: bool = False,
            **kwargs,
        ):
            return [
                {
                    "similarity_score_to_each_prototype": torch.randn(len(x), 4, 3, 5),
                    "logits": torch.randn(len(x), 2),
                }
                for i in range(2)
            ]

    return MockSTProtoPNet()


class MockBatchLoss:
    def __init__(self, device="cpu"):
        self.device = device

    def required_forward_results(self):
        return ["logits", "prototype_activations"]

    def __call__(self, target, metrics_dict, **kwargs):
        return torch.tensor(0.0, requires_grad=True)

    def to(self, device):
        self.device = device
        return self


class MockModelRegularization:
    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, model, metrics_dict):
        return torch.tensor(0.0, requires_grad=True)

    def to(self, device):
        self.device = device
        return self


class MockLoss(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.batch_loss = MockBatchLoss(device)
        self.model_regularization = MockModelRegularization(device)
        self.device = device

    def forward(
        self, target: torch.Tensor, model: "MockProtoPNet", metrics_dict: dict, **kwargs
    ):
        batch_loss = self.batch_loss(target=target, metrics_dict=metrics_dict, **kwargs)
        model_regularization = self.model_regularization(
            model, metrics_dict=metrics_dict
        )
        return batch_loss + model_regularization

    def to(self, device):
        super().to(device)
        self.batch_loss.to(device)
        self.model_regularization.to(device)
        self.device = device
        return self


@pytest.fixture
def mock_dataloaders():
    # Create mock dataset classes that return dictionaries
    class MockDataset:
        def __init__(self, num_samples):
            self.samples = [
                {
                    "img": torch.randn(1, 10),  # Batch size 1, 10 features
                    "target": torch.randint(0, 2, (1,)),  # Binary classification
                }
                for _ in range(num_samples)
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    return {
        "train": DataLoader(MockDataset(5), batch_size=1),
        "val": DataLoader(MockDataset(3), batch_size=1),
        "project": DataLoader(MockDataset(2), batch_size=1),
    }


@pytest.fixture
def mock_loss():
    return MockLoss()
