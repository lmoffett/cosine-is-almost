import os
import tempfile
from pathlib import Path

import pytest

import wandb


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


skip_if_not_wandb = pytest.mark.skipif(
    not wandb.api.api_key, reason="Weights & Biases login required"
)
