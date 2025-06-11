import re
from contextlib import contextmanager

import pytest
import torch
import torch.utils.model_zoo as model_zoo

from protopnet.pretrained import densenet_features, resnet_features
from protopnet.pretrained.google_drive import ensure_pretrained_model_from_google_drive


def is_valid_drive_id(drive_id: str) -> bool:
    # Google Drive file IDs are typically 33 characters long, consisting of letters, digits, dashes, and underscores.
    pattern = r"^[a-zA-Z0-9_-]{33}$"
    return bool(re.match(pattern, drive_id))


def validate_state_dict(state_dict):
    # 1. Check if it's a dictionary
    assert isinstance(state_dict, dict), "The state_dict is not a dictionary."

    # 2. Check keys and values
    for key, value in state_dict.items():
        assert isinstance(key, str), f"Key '{key}' is not a string."
        assert torch.is_tensor(value) or isinstance(
            value, (int, float, list, dict)
        ), f"Value for key '{key}' is not a valid tensor or compatible type."

    # 3. Ensure the state_dict is not empty
    assert len(state_dict) > 0, "The state_dict is empty."

    # 4. Check tensor integrity
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            assert (
                value.ndim >= 0
            ), f"Tensor for key '{key}' has an invalid number of dimensions."
            assert value.numel() > 0, f"Tensor for key '{key}' is empty."


@contextmanager
def state_dict_fs_validation(state_dict_path):
    """
    Context manager to handle state_dict validation.

    Args:
        state_dict_path (Path): Path to the state dictionary file.
        validate_function (callable): Function to validate the state dictionary.
    """
    # Remove the file if it exists before entering the block
    if state_dict_path.exists():
        state_dict_path.unlink()

    # Enter the block
    yield state_dict_path
    # After the block, ensure the state_dict_path exists
    assert (
        state_dict_path.exists()
    ), f"Expected state_dict file {state_dict_path} to exist after the block."
    # Validate the state dictionary
    validate_state_dict(torch.load(state_dict_path, map_location="cpu"))


@pytest.fixture
def pretrained_test_dir(temp_dir):
    return temp_dir / "pretrained-test"


def test_download_google_drive(pretrained_test_dir):
    # resnet50 is just an example here
    assert is_valid_drive_id(
        resnet_features.model_urls["resnet50[pretraining=inaturalist]"]
    )

    with state_dict_fs_validation(
        pretrained_test_dir / "gdrive-model.pth"
    ) as state_dict_path:
        ensure_pretrained_model_from_google_drive(
            resnet_features.model_urls["resnet50[pretraining=inaturalist]"],
            state_dict_path,
        )


def test_download_pytorch(pretrained_test_dir):
    # densenet169 is just an example here
    model_url = densenet_features.model_urls["densenet169"]
    download_path = pretrained_test_dir / "model-zoo"
    model_path = download_path / model_url.split("/")[-1]
    with state_dict_fs_validation(model_path):
        model_zoo.load_url(model_url, download_path)
