import logging

import click

log = logging.getLogger(__name__)


def common_training_options(f):
    """Common options for training commands."""
    decorators = [
        click.option(
            "--verify", is_flag=True, help="Train a single iteration of all phases"
        ),
        click.option(
            "--dry-run", is_flag=True, help="Setup a training run, but do not run it"
        ),
        click.option("--dataset", default="cub200", help="Dataset to use for training"),
        click.option(
            "--backbone",
            default="resnet50[pretraining=inaturalist]",
            help="Backbone to train with",
        ),
        click.option("--run-id", help="Custom id for the run"),
        click.option(
            "--activation-function",
            default="cosine",
            help="Activation function to use for training",
        ),
        click.option(
            "--save-on-proto-updates",
            help="Save model before and after each change to the prototypes",
            is_flag=True,
            default=False,
        ),
        click.option(
            "--phase-multiplier",
            default=1,
            help="Multiplier for the number of epochs in each phase",
        ),
        click.option(
            "--proto-channels",
            default=None,
            type=int,
            help="Number of channels in the prototype layer",
        ),
        click.option(
            "--fa-func",
            default="serial",
            help="Fine-annotation loss that will be calculated",
        ),
        click.option(
            "--fa-coef",
            type=float,
            default=None,
            help="Coefficient for fine-annotation loss",
        ),
    ]
    for decorator in reversed(decorators):
        f = decorator(f)
    return f
