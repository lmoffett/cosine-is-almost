import datetime
import itertools
import logging
import os
import re
import subprocess
import time
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    ItemsView,
    List,
    Optional,
    Tuple,
)

import click
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

import wandb

from ..py_dispatch import create_pydantic_model_for_method, load_function_from_path
from ..cli import EnvDefaultOption, GenericKeyValueArgs, load_yaml_config, process_runner_config
from .sweep_runners import SlurmWandBAgentRunner, SubprocessAgentRunner
from ..slurm.slurm_runner import SlurmConfig

log = logging.getLogger(__name__)


class ValuesSweepParameter(BaseModel):
    values: List[Any]


class WandBConfigExtended(BaseModel):
    description: Optional[str] = Field(default=None)
    method: str
    metric: Any
    gpu_time_limit: Optional[str] = Field(default=None)
    parameters: Optional[Dict[str, Any]] = Field(default=None)
    train_method: Optional[str] = Field(default=None)
    program: Optional[str] = Field(default=None)
    command: Optional[List[str]] = Field(default=None)
    hyper_sweep_parameters: Optional[Dict[str, ValuesSweepParameter]] = Field(
        default=None
    )
    preflight_parameters: Dict[str, Any] = Field(default=None)

    @model_validator(mode="after")
    def validate_gpu_time_limit(self) -> "WandBConfigExtended":
        if self.gpu_time_limit is not None:
            if not re.match(r"^\d+[m]$", self.gpu_time_limit):
                raise ValueError("gpu_time_limit must be in minutes (e.g., 100m)")
        return self

    @model_validator(mode="after")
    def validate_train_method_or_program_command(self) -> "WandBConfigExtended":
        if self.train_method is not None:
            if self.program is not None or self.command is not None:
                raise ValueError(
                    "If train_method is specified, program and command must not be specified"
                )
        elif self.program is None and self.command is None:
            raise ValueError(
                "Either train_method or both program and command must be specified"
            )
        elif (self.program is None) != (self.command is None):
            raise ValueError("program and command must be specified together")
        return self

    @model_validator(mode="after")
    def validate_sweep_parameters(self) -> "WandBConfigExtended":
        has_parameters = bool(self.parameters)  # None or {} will be False
        has_hyper_sweep = self.hyper_sweep_parameters is not None

        if not (has_parameters or has_hyper_sweep):
            raise ValueError(
                "At least one of parameters or hyper_sweep_parameters must be non-empty"
            )
        return self


def generate_sweep_combinations(
    sweep_parameters: dict, overrides: Optional[dict] = None
) -> list:
    """
    Generate the cross-product of all values in the sweep parameters, taking into account any overrides.

    Args:
        sweep_parameters (dict): A dictionary where each key has a nested dict with a 'values' list.
        overrides (Optional[dict]): A dictionary of parameter values that should override sweep parameters.
            Any parameter in overrides will not participate in the cross-product generation.

    Returns:
        list: A list of dictionaries representing the cross-product of non-overridden values,
                combined with the override values.
    """
    if overrides is None:
        overrides = {}
    if sweep_parameters is None:
        sweep_parameters = {}

    # Remove any parameters that are being overridden
    sweep_parameters = {k: v for k, v in sweep_parameters.items() if k not in overrides}

    if not sweep_parameters:
        return [overrides.copy()]

    # Extract keys and their corresponding values lists for remaining parameters
    keys = sweep_parameters.keys()
    values = [param["values"] for param in sweep_parameters.values()]

    # Compute the cross-product
    cross_product = itertools.product(*values)

    # Create dictionaries with both the cross-product values and overrides
    result = []
    for combination in cross_product:
        config = dict(zip(keys, combination))
        config.update(overrides)  # Add the override values
        result.append(config)

    return result


def shorten_key(s):
    return "".join([sub[:4] for sub in str(s).split("_")])


def shorten_bracket_content(value: Any) -> str:
    """
    Shorten the content inside square brackets to the first three characters
    on each side of the equals sign.

    Args:
        value (str): A string potentially containing content in square brackets.

    Returns:
        str: The modified string with the shortened bracket content.
    """
    # Regex to match content inside square brackets
    pattern = r"\[(.*?)\]"

    # Function to process the matched content
    def shorten_match(match):
        content = match.group(1)  # Extract the content inside the brackets
        if "=" in content:
            left, right = content.split("=", 1)  # Split into left and right parts
            shortened = (
                f"{left[:4]}={right[:4]}"  # Shorten each part to the first 3 characters
            )
            return f"[{shortened}]"
        return match.group(0)  # If no '=', return the match as is

    # Substitute the matched content with the shortened version
    return re.sub(pattern, shorten_match, str(value))


def create_sweep_name(base_name, sweep_params):
    """Create a sweep name and tags from base name and sweep-level parameters."""
    if not sweep_params or len(sweep_params) == 0:
        return base_name

    return (
        base_name
        + ":"
        + "&".join(
            f"{shorten_key(k)}={shorten_bracket_content(v)}"
            for k, v in sweep_params.items()
        )
    )


@click.group()
def sweep():
    """Group of commands related to W&B sweeps."""
    pass


def generate_sweep_configs(
    sweep_config: Dict,
    base_name: str,
    preflight: bool = False,
    overrides: Optional[Dict] = None,
) -> Tuple[Dict, List[Tuple[ItemsView, Dict]]]:
    """
    Generate sweep configurations from the base config and extensions.

    In preflight mode:
    - Every parameter must have a corresponding override in preflight_parameters
    - hyper_sweep_parameters are converted to regular parameters for grid search
    - An error is thrown if a preflight_parameter tries to override a hyper_sweep_parameter
    - The method is set to 'grid' for comprehensive testing

    Args:
        sweep_config (Dict): The base sweep configuration
        base_name (str): The base name for the sweep
        preflight (bool): Whether to generate preflight configurations
        overrides (Optional[Dict]): Optional parameter overrides (not used in preflight mode)

    Returns:
        Tuple[Dict, List[Tuple[ItemsView, Dict]]]: A tuple containing:
            - The base sweep configuration
            - A list of tuples, each containing:
                - The sweep-specific parameters as dict_items
                - The complete sweep configuration for this combination

    Raises:
        ValueError: If a preflight parameter conflicts with a hyper_sweep_parameter
        ValueError: If any parameter lacks a preflight override value
    """
    sweep_config = sweep_config.copy()

    sweep_configs = []

    # Extract special parameter types
    hyper_sweep_parameters = sweep_config.pop("hyper_sweep_parameters", None) or {}
    preflight_parameters = sweep_config.pop("preflight_parameters", None) or {}
    regular_parameters = sweep_config.get("parameters", {})
    gpu_time_limit = sweep_config.pop("gpu_time_limit", None)

    if preflight:
        # Check for conflicts between preflight parameters and hyper_sweep parameters
        conflicts = set(preflight_parameters.keys()) & set(
            hyper_sweep_parameters.keys()
        )
        if conflicts:
            raise ValueError(
                f"Preflight parameters cannot override hyper_sweep parameters. Conflicts: {conflicts}"
            )

        # Identify parameters that need preflight values (those with multiple values)
        params_needing_preflight = {
            param_name
            for param_name, param_config in regular_parameters.items()
            if len(param_config.get("values", [])) > 1
        }

        # Check that all required parameters have preflight values
        missing_preflight = params_needing_preflight - set(preflight_parameters.keys())
        if missing_preflight:
            raise ValueError(
                f"Parameters with multiple values must have preflight values. Missing preflight values for: {missing_preflight}"
            )

        # For single-value parameters, use their original value if not in preflight_parameters
        single_value_params = {
            param_name: param_config["values"][0]
            for param_name, param_config in regular_parameters.items()
            if len(param_config.get("values", [])) == 1
            and param_name not in preflight_parameters
        }

        # Add single-value parameters to preflight_parameters if they're not already there
        preflight_parameters = {**single_value_params, **preflight_parameters}

        # Override method to grid for comprehensive testing
        sweep_config["method"] = "grid"
        sweep_config["entity"] = "test"

        # Convert hyper_sweep_parameters to regular parameters format
        hyper_params_as_values = {
            k: {"values": v["values"]} for k, v in hyper_sweep_parameters.items()
        }

        # Convert preflight parameters to the correct format
        preflight_params_as_values = {
            k: {"values": [v]} for k, v in preflight_parameters.items()
        }

        # Combine all parameter types
        sweep_config["parameters"] = {
            **preflight_params_as_values,  # Regular parameters with their preflight values
            **hyper_params_as_values,  # Hyper sweep parameters for grid search
        }

        if overrides and len(overrides) > 0:
            new_parameters = {}
            for key, value in sweep_config["parameters"].items():
                if key in overrides:
                    # Handle both single values and sets in overrides
                    override_values = overrides[key]
                    if isinstance(override_values, (set, list)):
                        new_parameters[key] = {"values": list(override_values)}
                    else:
                        new_parameters[key] = {"values": [override_values]}
                else:
                    new_parameters[key] = value

            sweep_config["parameters"] = new_parameters

        hyper_parameters_with_multiple_values = {
            k for k, v in hyper_params_as_values.items() if len(v["values"]) > 1
        }
        if "train_method" in sweep_config:
            sweep_config["command"] = sweep_command(
                sweep_config.pop("train_method"),
                run_name_parameters=hyper_parameters_with_multiple_values,
                gpu_time_limit=gpu_time_limit,
            )

        # Generate a single sweep with all combinations
        sweep_name = f"{base_name}:preflight"
        sweep_config["name"] = sweep_name

        # Add description of parameters used
        param_description = (
            f"\n\n[preflight_parameters]\n{yaml.dump(preflight_parameters)}\n"
            f"[hyper_sweep_parameters]\n{yaml.dump(hyper_sweep_parameters)}"
        )
        sweep_config["description"] = (
            f"{sweep_config.get('description', '')}{param_description}"
        )

        sweep_configs.append((None, sweep_config))

    else:
        if "train_method" in sweep_config:
            sweep_config["command"] = sweep_command(
                sweep_config.pop("train_method"), gpu_time_limit=gpu_time_limit
            )
        # Regular sweep mode - handle overrides if provided
        if overrides and len(overrides) > 0:
            new_hyper_sweep_parameters = {}
            for key, value in hyper_sweep_parameters.items():
                if key in overrides:
                    # Handle both single values and sets in overrides
                    override_values = overrides[key]
                    if isinstance(override_values, (set, list)):
                        new_hyper_sweep_parameters[key] = {
                            "values": list(override_values)
                        }
                    else:
                        new_hyper_sweep_parameters[key] = {"values": [override_values]}
                else:
                    new_hyper_sweep_parameters[key] = value
            hyper_sweep_parameters = new_hyper_sweep_parameters

        # Generate combinations from hyper_sweep_parameters
        sweep_combinations = generate_sweep_combinations(hyper_sweep_parameters)

        for sweep_specific_params in sweep_combinations:
            current_config = sweep_config.copy()

            # Convert current combination to parameter format
            sweep_params_as_values = {
                k: {"values": [v]} for k, v in sweep_specific_params.items()
            }

            # Update parameters with current combination
            current_config["parameters"] = current_config.get("parameters", {}).copy()
            current_config["parameters"].update(sweep_params_as_values)

            # Update description
            current_config["description"] = (
                f"{current_config.get('description', '')}\n\n"
                f"[hyper_sweep_parameters]\n{yaml.dump(sweep_specific_params)}"
            )

            # Generate sweep name
            sweep_name = create_sweep_name(base_name, sweep_specific_params)
            current_config["name"] = sweep_name

            sweep_configs.append((dict(sweep_specific_params.items()), current_config))

    return sweep_config, sweep_configs


class SweepAction(Enum):
    PROCEED = "proceed"
    SKIP = "skip"
    WAIT = "wait"


def get_user_action(sweep_specific_params):
    """Prompt user for action on the sweep."""
    if sweep_specific_params is not None:
        click.echo(
            f"\nPreparing to launch sweep with hyper-sweep-parameters:\n{yaml.dump(sweep_specific_params)}"
        )
    else:
        click.echo("\nPreparing to launch sweep.")

    # Define valid inputs for each action
    action_map = {
        "p": SweepAction.PROCEED,
        "proceed": SweepAction.PROCEED,
        "s": SweepAction.SKIP,
        "skip": SweepAction.SKIP,
        "w": SweepAction.WAIT,
        "wait": SweepAction.WAIT,
    }

    while True:
        action_input = click.prompt(
            "Choose action (p)roceed/(s)kip/(w)ait", default="p"
        ).lower()

        if action_input in action_map:
            return action_map[action_input]

        click.echo("Invalid input. Please use p/proceed, s/skip, or w/wait")


def interactive_sweep_runner(
    config, sweep_specific_params, runner, interactive=True, wait_interval=20
):
    """
    Run sweeps with interactive controls.

    Args:
        config: The sweep configuration
        sweep_specific_params: Parameters specific to this sweep
        runner: The runner instance (subprocess or slurm)
        interactive: Whether to run in interactive mode
        wait_interval: Seconds to wait between status checks when in wait mode

    Returns:
        sweep_id if the sweep was created, None if skipped
    """
    while True:
        runner.check_runner_status()

        if not interactive:
            action = SweepAction.PROCEED
        else:
            action = get_user_action(sweep_specific_params)

        if action == SweepAction.PROCEED:
            sweep_id = wandb.sweep(config)
            if sweep_specific_params is not None:
                logging.info(
                    "Sweep created with ID: %s for hyper-sweep-parameters\n%s",
                    sweep_id,
                    yaml.dump(sweep_specific_params),
                )
            else:
                logging.info("Sweep created with ID: %s", sweep_id)

            runner.run_wandb_agents(sweep_id)
            return sweep_id

        elif action == SweepAction.SKIP:
            logging.info("Skipping sweep with parameters %s", sweep_specific_params)
            return None

        elif action == SweepAction.WAIT:
            logging.info("Waiting and checking runner status...")
            time.sleep(wait_interval)
            # Continue the while loop to prompt again


def tuples_to_dict_with_sets(tuple_list):
    """
    Convert a list of tuples to a dictionary where values for duplicate keys are combined into sets.

    Args:
        tuple_list: List of tuples where each tuple is (key, value)

    Returns:
        Dictionary where values for the same key are combined into sets
    """
    result = {}
    for key, value in tuple_list:
        if key in result:
            # If the value is already a set, add to it
            if isinstance(result[key], set):
                result[key].add(value)
            # If we have a duplicate key, convert to set
            else:
                result[key] = {result[key], value}
        else:
            # First occurrence of the key
            result[key] = value
    return result


@sweep.command("launch")
@click.argument(
    "experiment-config", type=click.Path(exists=True, readable=True), required=True
)
@click.option(
    "--wandb-entity", cls=EnvDefaultOption, envvar="WANDB_ENTITY", required=True
)
@click.option(
    "--wandb-project", cls=EnvDefaultOption, envvar="WANDB_PROJECT", required=True
)
@click.option(
    "--runner-config",
    cls=EnvDefaultOption,
    envvar="SWANDB_SWEEP_RUNNER_CONFIG",
    default="subprocess",
)
@click.option(
    "--runner-config-set",
    multiple=True,
    type=(str, str),
    help="Override runner config values. Can be specified multiple times. Example: --runner-config-set mem_gb 4",
)
@click.option(
    "--experiment-dir",
    cls=EnvDefaultOption,
    envvar="SWANDB_EXPERIMENT_BASE_DIR",
    default="experiments",
)
@click.option("--base-name", type=str, required=False)
@click.option("--preflight", is_flag=True, default=False)
@click.argument("fixed-hyped-sweep-parameters", nargs=-1, type=GenericKeyValueArgs())
@click.option("--auto-launch", is_flag=True, default=False)
def run(
    experiment_config,
    base_name,
    preflight,
    wandb_entity,
    wandb_project,
    runner_config,
    runner_config_set,
    fixed_hyped_sweep_parameters,
    auto_launch,
    experiment_dir,
):
    """
    Process two YAML files: wandb-sweep-config and sweep-extensions.

    \b
    Arguments:
    hyper-sweep-parameters: Weights and biases-like sweep configuration with additional hyperparameters.
    sweep-extensions: Path to the sweep-extensions YAML file.
    base-name: Base name for the sweep runs. Will be appended with sweep-level parameters.
    preflight: Run the preflight checks and exit.
    runner-config: Either [slurm, subprocess] or a path to a YAML file with runner configuration.
    fixed-hyped-sweep-parameters: Fixed hyperparameters to be used in all runs. Overrides hyper_sweep_parameters in the config. The values must be present in the config.
    auto-launch: do not interactively wait for confirmation to continue.
    experiment_dir: Path to the directory for the experiment outputs.
    """
    with open(experiment_config, "r") as f:
        sweep_config = yaml.safe_load(f)
        WandBConfigExtended.model_validate(sweep_config)

    runner_config = process_runner_config(runner_config, runner_config_set)
    experiment_dir = Path(experiment_dir)

    if preflight:
        experiment_dir = experiment_dir / "preflight"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        wandb_project = f"{wandb_project}--preflight"

    if runner_config["runner"] == "subprocess":
        runner = SubprocessAgentRunner()
    elif runner_config["runner"] == "slurm":
        runner = SlurmWandBAgentRunner(
            slurm_config=SlurmConfig.from_dict(runner_config), experiment_dir=experiment_dir
        )
    else:
        raise ValueError(f"Invalid runner config: {runner_config}")

    sweep_config["entity"] = wandb_entity
    sweep_config["project"] = wandb_project
    base_name = base_name if base_name else str(Path(experiment_config).stem)

    base_config, sweep_configs = generate_sweep_configs(
        sweep_config,
        base_name,
        preflight=preflight,
        overrides=tuples_to_dict_with_sets(fixed_hyped_sweep_parameters),
    )

    log.info(
        f"Launching {len(sweep_configs)} sweeps with base config:\n%s\n",
        ">  " + "\n>  ".join(yaml.dump(base_config).split("\n")[:-1]),
    )

    log.info(
        "Runner config:\n%s\n",
        ">  " + "\n>  ".join(yaml.dump(runner_config).split("\n")[:-1]),
    )

    launched_sweep = namedtuple(
        "LaunchedSweep", ["sweep_id", "sweep_specific_params", "url"]
    )
    launched_sweeps = []
    for sweep_specific_params, config in sweep_configs:
        sweep_id = interactive_sweep_runner(
            config, sweep_specific_params, runner, interactive=not auto_launch
        )
        url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/sweeps/{sweep_id}"
        log.info("View sweep at %s", url)
        launched_sweeps.append(launched_sweep(sweep_id, sweep_specific_params, url))

    runner.check_runner_status()
    log.info("%d sweeps launched:\n", len(launched_sweeps))

    for sweep in launched_sweeps:
        log.info(
            "%s [%s]: %s",
            sweep.sweep_id,
            (
                "|".join([f"{k}={v}" for k, v in sweep.sweep_specific_params.items()])
                if sweep.sweep_specific_params
                else ""
            ),
            sweep.url,
        )


def sweep_command(train_method, run_name_parameters=None, gpu_time_limit=None):
    command = [
        "python",
        "-m",
        "swandb",
        "sweep",
        "train",
        train_method,
        "${args_no_hyphens}",
    ]

    if run_name_parameters:
        command.append(f"--run-name-parameters={','.join(run_name_parameters)}")

    if gpu_time_limit:
        # Remove the 'm' from the end of the time limit
        # This is enforced through pydantic
        command.append(f"--gpu-time-limit={gpu_time_limit[:-1]}")

    return command

def guard_sweep_runtime(sweep_full_id, gpu_time_limit):
    log.info(f"Checking runtime for sweep: {sweep_full_id}")

    log.debug(f"max runtime for sweep is: {gpu_time_limit}")

    gpu_time_limit_seconds = int(gpu_time_limit) * 60

    total_runtime = 0
    api = wandb.Api()
    runs = api.sweep(sweep_full_id).runs
    for existing_run in runs:
        if existing_run.state in ["finished", "crashed", "failed"]:
            if "actual_runtime" in existing_run.summary:
                runtime = existing_run.summary["actual_runtime"]
                total_runtime += runtime
            else:
                log.error(f"{existing_run} has no runtime info")

        else:
            # check if the run has actually started
            # a run could have been in the system as "running"
            # but not actually started running (nothing logged)
            if hasattr(existing_run, "start_time"):
                start_time_unix = existing_run.start_time
                start_time = datetime.fromtimestamp(start_time_unix)
                elapsed_time = datetime.now() - start_time
                elapsed_seconds = elapsed_time.total_seconds()
                total_runtime += elapsed_seconds
            else:
                log.error(f"{existing_run} has no start time")

    log.info(
        f"Total runtime elapsed in sweep is {int(total_runtime//60)}m out of {gpu_time_limit}m allotted"
    )

    if total_runtime >= gpu_time_limit_seconds:
        log.info("[-] Total runtime exceeded, not starting a new run.")
        subprocess.run(["wandb", "sweep", "--cancel", sweep_full_id])
        raise SystemExit("Total runtime exceeded, not starting a new run.")


def format_run_name(sweep_id, run_number, extra_args, run_name_parameters):
    run_name = f"{run_number}@{sweep_id}"

    if run_name_parameters and len(run_name_parameters) > 0:
        run_name_parameters = run_name_parameters.split(",")

        assert all(
            p in extra_args for p in run_name_parameters
        ), f"Run name parameters {run_name_parameters} must be present in arguments {extra_args}"

        run_name = (
            run_name + "-" + "-".join([f"{extra_args[p]}" for p in run_name_parameters])
        )

    return run_name


@sweep.command("train")
@click.argument("train-method", type=str)
@click.argument("train-args", nargs=-1, type=GenericKeyValueArgs())
@click.option("--train-args-file", type=click.Path(exists=True, readable=True))
@click.option("--gpu-time-limit", type=float, default=float("inf"))
@click.option("--run-name-parameters", type=str, default=None)
def sweep_train(
    train_method, train_args, train_args_file, gpu_time_limit, run_name_parameters
):
    """
    Train a model using a specified method.

    train-runner is the dotted path to a Python method (e.g., `model.trainer.run`).
    train-args are additional key=value arguments for the launcher method. These take precedence over the file.
    train-args-file is a YAML file containing the key-value arguments.
    gpu-time-limit is the maximum runtime allowed for all runs in the sweep in minutes. Any run starting after this time will be terminated.
    run-name-parameters is a comma-separated list of parameters to include in the run name in order.
    """
    if train_args_file:
        file_train_args = load_yaml_config(train_args_file)
        log.debug(f"Loaded training arguments from file:\n{file_train_args}")
    else:
        file_train_args = {}

    cli_train_args = dict(train_args)

    train_args = {**file_train_args, **cli_train_args}

    train_method_handle = load_function_from_path(train_method)

    # Validate the arguments using the Pydantic model
    try:
        Model = create_pydantic_model_for_method(train_method_handle)
        # validates the args and casts them to the right types
        validated_args = Model(**train_args)
    except ValidationError as e:
        log.error(
            f"Extra training arguments do not match the argument signature of {train_method}: {e}"
        )
        raise e

    log.info(f"Calling {train_method} with arguments: {train_args}")

    wandb.init()

    if wandb.run.sweep_id is None:
        sweep_full_id = None
        log.warning(
            "Running outside of sweep. Continuing without sweep preprocessing. This should only be done for debugging."
        )
    else:
        sweep_id = wandb.run.sweep_id
        sweep_full_id = f"{wandb.run.entity}/{wandb.run.project}/{sweep_id}"
        run_number_in_sweep = wandb.run.name.split("-")[-1]
        wandb.run.name = format_run_name(
            sweep_id, run_number_in_sweep, train_args, run_name_parameters
        )

        # setup in-sweep logging
        if os.environ.get("SWANDB_LOG_DIR"):
            log_dir = Path(os.environ.get("SWANDB_LOG_DIR"))
            log_path = log_dir / f"{wandb.run.name}.log"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.getLogger().level)
            file_handler.setFormatter(logging.getLogger().handlers[0].formatter)
            logging.getLogger().addHandler(file_handler)
            log.info("Logging to %s", log_path)

        if gpu_time_limit != float("inf"):
            guard_sweep_runtime(sweep_full_id, int(gpu_time_limit))

    start_time = datetime.datetime.now()

    try:
        train_method_handle(**validated_args.model_dump())

    finally:
        elapsed_time = datetime.datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()

        log.info(f"This run elapsed_time {elapsed_seconds}s")
        wandb.log({"actual_runtime": elapsed_seconds})
        wandb.run.summary["actual_runtime"] = elapsed_seconds

        wandb.finish()

        if gpu_time_limit != float("inf") and sweep_full_id is not None:
            guard_sweep_runtime(sweep_full_id, int(gpu_time_limit))
