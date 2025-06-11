import logging
import os
import random
import shlex
import sys

import click
import numpy as np
import torch
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)


def preprocess_args(args):
    # Look for space-separated arguments and split them
    new_args = []
    for arg in args:
        if " " in arg:
            # shlex.split handles quotes and escapes properly
            new_args.extend(shlex.split(arg))
        else:
            new_args.append(arg)
    return new_args


def setup_reproducibility(seed):
    """Set up reproducibility settings."""
    log.info(f"Using seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.multiprocessing.set_start_method("spawn", force=True)
    np.random.seed(seed)
    random.seed(seed)


def get_default_seed():
    try:
        return int(os.environ.get("PPNXT_SEED", 1234))
    except ValueError:
        raise click.BadParameter(
            "PPNXT_SEED must be an integer, or leave it unset for the default (1234)."
        )


def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise click.BadParameter(f"Invalid log level: {log_level}")
    logging.getLogger().setLevel(numeric_level)
    log.info(f"Logging initialized at {log_level.upper()} level")


@click.group()
@click.option("--seed", type=int, default=get_default_seed)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
)
def cli(seed, log_level):
    """ProtoPNet CLI - A tool for running ProtoPNet experiments."""
    # Initialize logging
    setup_logging(log_level)
    setup_reproducibility(seed)


# Import commands AFTER the CLI is defined
from . import (  # noqa: E402
    eval,
    pacmap,
    train_deformable,
    train_protopnet,
    train_prototree,
    train_support_trivial,
    train_tesnet,
    visualization,
)
from .datasets import cli as datasets_cli  # noqa: E402

# Add each module's commands to the main CLI
cli.add_command(train_protopnet.run)
cli.add_command(train_deformable.run)
cli.add_command(train_tesnet.run)
cli.add_command(train_prototree.run)
cli.add_command(train_support_trivial.run)
cli.add_command(visualization.run)
cli.add_command(pacmap.run)
cli.add_command(eval.run)
cli.add_command(datasets_cli.datasets)

if __name__ == "__main__":
    sys.argv[1:] = preprocess_args(sys.argv[1:])
    with logging_redirect_tqdm():
        cli()
