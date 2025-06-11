# To debug these tests, the easiest thing is to run them from the command line.
# Just take the string from the call to python() below and run it from the command line as 'python ' + cmd
# If you need to debug, you can use your IDE debugger but stil running from the base command.

import os
import random
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch


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


@pytest.fixture(scope="session")
def cuda_available():
    return torch.cuda.is_available()


# Register the condition globally
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("Skipping test because CUDA is not available.")


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


def assert_epoch_training(
    stdout: str,
    completed_epochs: List[str],
):
    # Training schedule looks like this:
    # TrainingSchedule(
    #   Phase(warm, eval=eval, duration=1)
    #   Phase(joint, eval=eval, duration=1)
    #   Phase(project, eval=eval, duration=1)
    #   Phase(last_only, eval=eval, duration=1)
    #   Phase(prune_prototypes, eval=eval, duration=1)
    # )

    pattern_lines = [
        "TrainingSchedule(",
    ]
    for i, epoch in enumerate(completed_epochs):
        if epoch == "project":
            pattern_lines.append("Phase(project, eval=eval, duration=1)")
        elif epoch == "prune":
            pattern_lines.append("Phase(prune_prototypes, eval=eval, duration=1)")
        else:
            pattern_lines.append(f"Phase({epoch}, eval=eval, duration=1)")

    pattern_lines[-1] = pattern_lines[-1][:-1]
    pattern_lines.append(")")
    remaining_lines = stdout
    for line in pattern_lines:
        assert line in remaining_lines, f"Expected line not found in output: {line}"
        remaining_lines = remaining_lines.split(line, 1)[1]
    assert (
        f"Training complete after {len(completed_epochs)} epochs" in stdout
    ), "Training did not complete successfully"


@pytest.fixture(scope="package")
def cifar10_squeezenet1_0_path(temp_root_dir):
    run_id = "test_cifar10_squeezenet1_0"
    run_dir = temp_root_dir.resolve() / "inttest" / "artifacts" / "runs" / run_id

    retrain = os.environ.get("PPNXT_TEST_TRAIN_INT_MODEL", "True").lower() == "true"
    if retrain:
        stdout, _ = python(
            f"-u -m protopnet train-protopnet --dataset=cifar10 --backbone=squeezenet1_0 --verify --run-id={run_id} --proto-channels=32",
            {
                "WANDB_MODE": "dryrun",
                "PPNXT_ARTIFACT_DIR": str(temp_root_dir / "inttest" / "artifacts"),
            },
        )
        assert_epoch_training(
            stdout, ["warm", "joint", "project", "last_only", "prune"]
        )

    for file in run_dir.glob(f"{run_id}_best@*_last_only_*.pth"):
        return file


@contextmanager
def disable_exception_traceback():
    """
    All traceback information is suppressed and only the exception type and value are printed
    """
    default_value = getattr(
        sys, "tracebacklimit", 1000
    )  # `1000` is a Python's default value
    sys.tracebacklimit = 0
    yield
    sys.tracebacklimit = default_value


def print_output(stdout, stderr):
    output = ""
    output += "STDOUT:\n"
    for line in stdout.splitlines():
        output += "> " + line + "\n"
    output += "\n"
    output += "STDERR:" + "\n"
    for line in stderr.splitlines():
        output += "> " + line + "\n"
    return output


def python(cmd, extra_env={}):
    env = os.environ.copy()
    extra_env = {
        "DISABLE_TQDM": "1",
        "WANDB_MODE": "dryrun",
        **extra_env,
    }
    env.update(extra_env)
    """Executes a command, captures stdout and stderr, and raises an error if it returns a non-0 exit code."""
    try:
        # If you want the command output as a string, use subprocess.PIPE as the stdout and stderr value
        result = subprocess.run(
            sys.executable + " " + cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            env=env,
        )
        print(print_output(result.stdout, result.stderr))

        return result.stdout, result.stderr

    except subprocess.CalledProcessError as e:
        # Suppress the runtime exception since we care about the subprocess exception and it's already been printed
        pytest.fail(
            reason=f"Command '{cmd}' returned non-zero exit status {e.returncode}:\n\n{print_output(e.output, e.stderr)}",
            pytrace=False,
        )
