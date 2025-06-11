import os
from pathlib import Path
from typing import Any

import click
import yaml


class GenericKeyValueArgs(click.ParamType):
    """Custom type to parse key=value pairs into a dictionary."""

    name = "key_value_args"

    def convert(self, value, param, ctx):
        if "=" not in value:
            self.fail(
                f"Invalid argument format: {value}. Expected key=value.", param, ctx
            )
        key, val = value.split("=", 1)
        return key, val


class EnvDefaultOption(click.Option):
    """
    A Click option that uses an environment variable as the default value.
    Raises an error if neither the flag nor the environment variable is set.
    """

    def __init__(self, *args, envvar=None, **kwargs):
        self.required_message = kwargs.pop(
            "required_message",
            f"One of the flag or environment variable ({envvar}) must be set.",
        )
        super().__init__(*args, **kwargs)
        self.envvar = envvar

    def get_default(self, ctx: click.Context):
        # Use the environment variable if it's set
        if self.envvar and self.envvar in os.environ:
            return os.environ[self.envvar]

        # Otherwise, return the default value (if specified)
        return super().get_default(ctx)

    def process_value(self, ctx: click.Context, value: Any):
        if value is None:
            raise click.BadParameter(self.required_message)
        return value

def nested_set(dict_obj, keys, value):
    """Set a value in a nested dictionary using a list of keys."""
    current = dict_obj
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value

def load_yaml_config(file_path, overrides=None):
    """
    Load the YAML configuration file and apply any overrides.

    Args:
        file_path: Path to YAML config file
        overrides: List of tuples [(key, value), ...] where key can be dot-notated
    """
    # Load base configuration
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides if any exist
    if overrides:
        for key, value in overrides:
            # Split the key into parts for nested access
            key_parts = key.split(".")
            nested_set(config, key_parts, value)

    return config

def process_runner_config(runner_config, runner_config_set):
    if runner_config is None:
        return "subprocess"

    try:
        launcher_config_path = Path(runner_config)
        if not launcher_config_path.exists():
            raise ValueError(f"Launcher config file not found: {runner_config}")

        runner_config = load_yaml_config(
            launcher_config_path, overrides=runner_config_set
        )

        if "runner" not in runner_config:
            raise ValueError(
                "Invalid runner config. Field 'runner' is required to be one of ['subprocess', 'slurm']"
            )
        return runner_config
    except ValueError:
        if runner_config not in ["subprocess", "slurm"]:
            raise ValueError(
                "Invalid runner config. Field 'runner' is required to be one of ['subprocess', 'slurm']"
            )
        return {"runner": runner_config}