import logging
import os
import random
import shlex
import sys

import click
import numpy as np
import torch

if sys.stdin.isatty():
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )
else:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
    )

log = logging.getLogger(__name__)


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


def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise click.BadParameter(f"Invalid log level: {log_level}")
    logging.getLogger().setLevel(numeric_level)


def get_default_seed():
    try:
        return int(os.environ.get("SEED", 1234))
    except ValueError:
        raise click.BadParameter(
            "SEED must be an integer, or leave it unset for the default (1234)."
        )


def setup_reproducibility(seed):
    """Set up reproducibility settings."""
    log.info(f"Using seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.multiprocessing.set_start_method("spawn", force=True)
    np.random.seed(seed)
    random.seed(seed)


@click.group()
@click.option("--seed", type=int, default=get_default_seed)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
)
def cli(seed, log_level):
    """swandb - run weights and biases sweeps on SLURM."""
    setup_logging(log_level)
    setup_reproducibility(seed)


from .weights_and_biases.sweeps import (  # noqa E402 - Import at the end to avoid circular imports
    sweep,
)
from .weights_and_biases.reporting import (  # noqa E402 - Import at the end to avoid circular imports
    reporting,
)
from .weights_and_biases.runs import (  # noqa E402 - Import at the end to avoid circular imports
    runs,
)

cli.add_command(sweep)
cli.add_command(reporting)
cli.add_command(runs)

if __name__ == "__main__":
    sys.argv[1:] = preprocess_args(sys.argv[1:])
    cli()
