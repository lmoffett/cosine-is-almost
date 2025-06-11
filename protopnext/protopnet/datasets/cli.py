from typing import Optional

import click

from . import cars, cars_cropped, cub200_corners, cub200_cropped, dataset_prep, dogs


@click.group()
def datasets():
    """Dataset preparation and manipulation commands."""
    pass


@datasets.command("prep-metadata")
@click.argument("directory", type=click.Path(exists=True))
@click.option(
    "--dataset",
    type=click.Choice(
        ["cars", "dogs", "dogs-cropped", "cub200-cropped", "cars-cropped"],
        case_sensitive=False,
    ),
    required=True,
    help="Dataset to prepare metadata for",
)
def prep_metadata(directory: str, dataset: str):
    """Prepare metadata for a specific dataset."""
    if dataset == "cub200-cropped":
        cub200_cropped.crop_cub200(directory)
    elif dataset == "cars":
        cars.parse_cars_metadata(directory)
    elif dataset == "cars-cropped":
        cars_cropped.crop_cars(directory)
    elif dataset == "dogs":
        dogs.parse_dogs_metadata(directory)
    elif dataset == "dogs-cropped":
        dogs.crop_dogs(directory)
    else:
        raise ValueError(f"Unknown metadata action: {dataset}")


@datasets.command("create-splits")
@click.argument("base_dir", type=click.Path(exists=True))
@click.option("--image-dir", help="Name of image directory")
@click.option("--train-dir", help="Path to train directory")
@click.option("--test-dir", help="Path to test directory")
@click.option("--val-dir", help="Path to validation directory")
@click.option(
    "--cast-ids-to-int", is_flag=True, help="Treat image identifiers as integers"
)
def create_splits(
    base_dir: str,
    image_dir: Optional[str],
    train_dir: Optional[str],
    test_dir: Optional[str],
    val_dir: Optional[str],
    cast_ids_to_int: bool,
):
    """Create train/test/validation splits for a dataset."""
    # Filter out None values to match original behavior of argparse.SUPPRESS
    kwargs = {k: v for k, v in locals().items() if k != "base_dir" and v is not None}
    dataset_prep.create_splits(base_dir, **kwargs)


@datasets.command("crop-corners")
@click.option(
    "--base-dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to dataset directory",
)
@click.option("--image-dir", required=True, help="Name of image directory")
@click.option("--output-dir", required=True, help="Path to output directory")
def crop_corners(base_dir: str, image_dir: str, output_dir: str):
    """Crop corners from images in a dataset."""
    cub200_corners.crop_corners(
        base_dir=base_dir, image_dir=image_dir, output_dir=output_dir
    )
