import pathlib

import pytest
from PIL import Image

from .conftest import python


@pytest.fixture(scope="module")
def analysis_dir(temp_dir):
    return temp_dir / "analysis"


def test_render_prototypes(analysis_dir, cifar10_squeezenet1_0_path):
    stdout, _ = python(
        f"-u -m protopnet viz render-prototypes --model-path={cifar10_squeezenet1_0_path} --dataset=cifar10 --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed rendering of prototypes" in stdout


def test_local_analysis(analysis_dir, cifar10_squeezenet1_0_path):
    stdout, _ = python(
        f"-u -m protopnet viz local --model-path={cifar10_squeezenet1_0_path} --dataset=cifar10 --sample=6 --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed local analysis." in stdout


@pytest.mark.cuda
def test_pacmap(analysis_dir, cifar10_squeezenet1_0_path):
    stdout, _ = python(
        f"-u -m protopnet pacmap --model-path={cifar10_squeezenet1_0_path} --dataset=cifar10 --n-neighbors=1 --save-dir={analysis_dir} --sample=1",
        {"WANDB_MODE": "dryrun"},
    )
    assert "PaCMAP plotted" in stdout

    pacmap_path = pathlib.Path(analysis_dir / "pacmap.png")
    assert pacmap_path.exists()

    pacmap_image = Image.open(pacmap_path)
    assert isinstance(pacmap_image, Image.Image)
    assert pacmap_image.format == "PNG"
    assert pacmap_image.mode == "RGBA"

    pacmap_image.load()


@pytest.mark.cuda
def test_global_analysis(analysis_dir, cifar10_squeezenet1_0_path):
    stdout, _ = python(
        f"-u -m protopnet viz global --model-path={cifar10_squeezenet1_0_path} --dataset=cifar10 --sample=1 --output-dir={analysis_dir}",
        {"WANDB_MODE": "dryrun"},
    )
    assert "Completed global analysis." in stdout
