import inspect
import pathlib
import click

import pandas as pd
import logging
import click
import re

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Union

from .runs_runners import CommandGeneratorType, SlurmForeachRunRunner, SubprocessForeachRunRunner
from ..py_dispatch import create_pydantic_model_for_method, load_function_from_path
from ..cli import EnvDefaultOption, load_yaml_config, process_runner_config
from ..slurm.slurm_runner import SlurmConfig, SlurmRunner


log = logging.getLogger(__name__)


def create_template_command_generator(command_template: List[str]) -> CommandGeneratorType:
    """
    Create a command generator function from a command template.
    
    The template can contain placeholders:
    - ${grouping_key}: The value of the grouping key for this group
    - ${runs.X}: Comma-separated list of values for column X for all runs in the group
    
    Args:
        command_template: List of command arguments with placeholders
        
    Returns:
        A command generator function
    """
    def template_command_generator(runs_df: pd.DataFrame, array_mapping: Dict[str, str], grouping_key: str) -> Dict[int, List[str]]:
        commands = {}

        for array_idx, group_id in array_mapping.items():
            # Get rows for this group
            group_df = runs_df[runs_df[grouping_key] == group_id]
            
            # Process each argument in the template
            processed_args = []
            for arg in command_template:
                # Replace ${grouping_key} with the actual group ID
                if "${grouping_key}" in arg:
                    arg = arg.replace("${grouping_key}", str(group_id))
                
                # Replace ${runs.X} placeholders with comma-separated values
                for match in re.finditer(r'\${runs\.([^}]+)}', arg):
                    col_name = match.group(1)
                    if col_name in group_df.columns:
                        values = ",".join(str(v) for v in group_df[col_name].tolist())
                        arg = arg.replace(match.group(0), values)
                    else:
                        log.warning(f"Column '{col_name}' not found in data")
                        arg = arg.replace(match.group(0), "")
                
                processed_args.append(arg)
            
            commands[array_idx] = processed_args
        
        return commands
    
    return template_command_generator


def validate_command_generator(generator_func: Callable) -> None:
    """
    Validate that a command generator function has the correct signature.
    
    Args:
        generator_func: The command generator function to validate
        
    Raises:
        ValueError: If the function doesn't have the expected signature
    """
    # Create a model for the function
    try:
        model = create_pydantic_model_for_method(generator_func)
        
        # Check if the function has the required parameters
        params = list(inspect.signature(generator_func).parameters.keys())
        required_params = ["runs_df", "array_mapping", "grouping_key"]
        
        for param in required_params:
            if param not in params:
                raise ValueError(f"Command generator function missing required parameter: {param}")
        
        # Return the function if valid
        return generator_func
    except Exception as e:
        raise ValueError(f"Invalid command generator function: {e}")


def process_yaml_script_runner_config(
    config_file: str
) -> Dict[str, Any]:
    """
    Process a YAML configuration file for run_over_runs.
    
    Args:
        config_file: Path to the YAML configuration file
        
    Returns:
        Dictionary with processed configuration
    """
    result = {
        'grouping_key': None,
        'command_generator': None
    }
    
    try:
        # Load the config file
        config = load_yaml_config(config_file)
        
        # Extract grouping key
        if 'grouping_key' in config:
            result['grouping_key'] = config['grouping_key']
            log.info(f"Using grouping key from config: {result['grouping_key']}")
        
        # Extract command generator or command template
        if 'command_generator' in config:
            generator_path = config['command_generator']
            log.info(f"Loading command generator from: {generator_path}")
            
            try:
                # Load the function
                generator_func = load_function_from_path(generator_path)
                
                # Validate it has the correct signature
                result['command_generator'] = validate_command_generator(generator_func)
                
                log.info(f"Successfully loaded command generator")
            except Exception as e:
                log.error(f"Failed to load command generator: {e}")
                raise
                
        elif 'command' in config:
            if not isinstance(config['command'], list):
                raise ValueError(f"'command' must be a list of strings, got {type(config['command'])}")
            
            log.info(f"Creating command generator from template: {config['command']}")
            result['command_generator'] = create_template_command_generator(config['command'])
        
    except Exception as e:
        log.error(f"Error processing YAML config {config_file}: {e}")
        raise
    
    return result


def prepare_directories(experiment_dir: str) -> Tuple[Path, Path, Path]:
    """
    Create necessary directories for the experiment.
    
    Args:
        experiment_dir: Base directory for experiments
        
    Returns:
        Tuple of (experiment_dir, script_dir, log_dir) as Path objects
    """
    experiment_path = Path(experiment_dir)
    script_dir = experiment_path / "scripts"
    log_dir = experiment_path / "logs"
    
    for dir_path in [experiment_path, script_dir, log_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
        log.debug(f"Created directory: {dir_path}")
    
    return experiment_path, script_dir, log_dir


def get_runner(runner_config, experiment_dir):
    """
    Create appropriate runner based on configuration.
    
    Args:
        runner_config: Runner configuration dictionary
        experiment_dir: Base directory for experiments
        
    Returns:
        Appropriate runner object
    """
    if runner_config["runner"] == "subprocess":
        return None
    elif runner_config["runner"] == "slurm":
        runner = SlurmRunner(
            config=SlurmConfig.from_dict(runner_config)
        )
    else:
        raise ValueError(f"Invalid runner config: {runner_config}")
    
    return runner


def display_job_information(
    merged_df: pd.DataFrame, 
    runner_config: Dict[str, Any]
) -> bool:
    """
    Display job information and ask for confirmation.
    
    Args:
        merged_df: DataFrame with merged run and sweep data
        runner_config: Runner configuration
        
    Returns:
        True if user confirms, False otherwise
    """
    # Display sweep information
    sweep_groups = merged_df.groupby('sweep_id')
    log.info("=" * 60)
    log.info("Sweeps to be processed:")
    log.info("-" * 60)
    
    for sweep_id, group in sweep_groups:
        run_count = len(group)
        log.info(f"Sweep {sweep_id}: {run_count} runs")
    
    # Display runner information
    log.info("=" * 60)
    log.info("Runner configuration:")
    log.info("-" * 60)
    
    runner_type = runner_config.get("runner", "unknown")
    log.info(f"Runner type: {runner_type}")
    
    if runner_type == "slurm":
        log.info("SLURM configuration:")
        for key, value in runner_config.items():
            if key != "runner" and value is not None:
                log.info(f"  {key}: {value}")
    
    # Ask for confirmation
    log.info("=" * 60)
    confirm = input("Do you want to proceed with the jobs? [y/N]: ")
    return confirm.lower() in ['y', 'yes']

def process_sweep_jobs(
    merged_df: pd.DataFrame,
    runner,
    script_dir: Path,
    log_dir: Path
) -> List[str]:
    """
    Process and submit jobs for each sweep.
    
    Args:
        merged_df: DataFrame with merged run and sweep data
        runner: Runner object for job submission
        script_dir: Directory for job scripts
        log_dir: Directory for job logs
        
    Returns:
        List of submitted job IDs
    """
    sweep_groups = merged_df.groupby('sweep_id')
    all_job_ids = []
    
    for sweep_id, group in sweep_groups:
        # Get run IDs for this sweep
        run_ids = group['run_id'].tolist()
        
        # Generate command - mock for now
        run_ids_str = ','.join(map(str, run_ids))
        command = f"echo 'Processing sweep {sweep_id} with runs: {run_ids_str}'"
        
        # Create a unique job ID for this sweep
        job_id = f"sweep_{sweep_id}"
        
        # Submit the job
        try:
            log.info(f"Submitting job for sweep {sweep_id} with {len(run_ids)} runs")
            job_ids = runner.run_command(
                command=command,
                job_id=job_id,
                script_dir=script_dir,
                log_dir=log_dir
            )
            log.info(f"Submitted job for sweep {sweep_id}: {', '.join(job_ids)}")
            all_job_ids.extend(job_ids)
        except Exception as e:
            log.error(f"Failed to submit job for sweep {sweep_id}: {e}")
            raise
    
    return all_job_ids


@click.group('runs')
def runs():
    pass

@runs.command("foreach")
@click.argument(
    "command-config-file",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.argument(
    "runs-csv",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--runner-config",
    cls=EnvDefaultOption,
    envvar="SWANDB_RUNNER_CONFIG",
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
def runs_foreach(
    command_config_file: Union[str, pathlib.Path],
    runs_csv: Union[str, pathlib.Path], 
    runner_config: str, 
    runner_config_set: List[Tuple[str, str]], 
    experiment_dir: str,
):
    """
    Invoke a process over a set of runs grouped by a key.
    
    Args:
        command_config_file: Path to YAML configuration file specifying grouping_key and either 
                    command_generator or command template 
        runs_csv: Path to the CSV containing run data. Each run should have a unique ID in the column `run_id`.
        runner_config: Path to runner configuration file or configuration name
        runner_config_set: List of (key, value) pairs to override runner configuration
        experiment_dir: Base directory for experiments
    """
    runs_df = pd.read_csv(runs_csv)

    if len(runs_df) == 0:
        log.error("No runs found in the CSV file.")
        return

    assert runs_df['run_id'].nunique() == len(runs_df), "run_id must be unique in the CSV"
    
    try:
        command_config = process_yaml_script_runner_config(command_config_file)

        grouping_key = command_config['grouping_key']
        command_generator = command_config['command_generator']

    except Exception as e:
        log.error(f"Failed to process config file {command_config_file}: {e}")
        raise
    
    runner_config = process_runner_config(runner_config, runner_config_set)

    maybe_slurm_runner = get_runner(runner_config, experiment_dir)

    # FIXME - get_runner makes no sense in this context. It's a half-factory
    if maybe_slurm_runner is None:
        for_each_runner = SubprocessForeachRunRunner(command_generator, grouping_key=grouping_key)
    else:
        for_each_runner = SlurmForeachRunRunner(maybe_slurm_runner, command_generator, grouping_key=grouping_key)

    # Display job information and ask for confirmation
    if not for_each_runner.display_job_information(runs_df, command_config):
        log.info("Operation cancelled by user.")
        return
    
    for_each_runner.process_job_array(runs_df, )
