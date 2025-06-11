import logging
import os
import pandas as pd
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Callable

from ..slurm.slurm_runner import SlurmRunner

log = logging.getLogger(__name__)

CommandGeneratorType = Callable[[pd.DataFrame, Dict[int, Any], str], Dict[int, List[str]]]


class BaseForeachRunRunner:
    """
    Base runner for processing groups of runs.
    """
    
    def __init__(
        self, 
        command_generator: CommandGeneratorType, 
        grouping_key: str = None, 
        experiment_dir: Union[str, Path] = Path("experiments")
    ):
        """
        Initialize the runner.
        
        Args:
            command_generator: Function that generates commands for each group
            grouping_key: The column to group runs by (default: "run_id")
            experiment_dir: Base directory for experiments
        """
        self.command_generator = command_generator
        self.grouping_key = grouping_key if grouping_key is not None else 'run_id'
        self.experiment_dir = Path(experiment_dir)

        self.log_dir = self.experiment_dir / "logs"
        self.script_dir = self.experiment_dir / "scripts"
    
    def _prepare_directories(self) -> Tuple[Path, Path, Path]:
        """
        Create necessary directories for the experiment.
        
        Returns:
            Tuple of (experiment_dir, script_dir, log_dir) as Path objects
        """
        self.script_dir = self.experiment_dir / "scripts"
        self.log_dir = self.experiment_dir / "logs"
        
        for dir_path in [self.experiment_dir, self.script_dir, self.log_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
            log.debug(f"Created directory: {dir_path}")
        
        return self.experiment_dir, self.script_dir, self.log_dir
    
    def _create_array_mapping(self, runs_df: pd.DataFrame) -> Dict[int, Any]:
        """
        Create a mapping from array indices to group IDs.
        
        Args:
            runs_df: DataFrame with run data
            
        Returns:
            Dictionary mapping array indices to group IDs
        """
        group_ids = runs_df[self.grouping_key].unique()
        return {i: group_id for i, group_id in enumerate(group_ids)}
    
    def display_job_information(
        self,
        runs_df: pd.DataFrame,
        runner_config: Dict[str, Any]
    ) -> bool:
        """
        Display job information and ask for confirmation.
        
        Args:
            runs_df: DataFrame with run data
            runner_config: Runner configuration
            
        Returns:
            True if user confirms, False otherwise
        """
        # Get group information
        group_counts = runs_df.groupby(self.grouping_key).size()

        array_mapping = self._create_array_mapping(runs_df)
        
        log.info("=" * 60)
        log.info(f"Groups to be processed (using {self.grouping_key}):")
        log.info("-" * 60)
        
        for group_id, count in group_counts.items():
            log.info(f"{self.grouping_key} {group_id}: {count} runs")
        
        # Display array mapping
        log.info("=" * 60)
        log.info("Job Array Mapping:")
        log.info("-" * 60)
        log.info(f"Array Index -> {self.grouping_key}")
        
        for array_idx, group_id in array_mapping.items():
            log.info(f"    {array_idx} -> {group_id}")
        
        # Display runner information
        log.info("=" * 60)
        log.info("Runner configuration:")
        log.info("-" * 60)
        
        runner_type = runner_config.get("runner", "unknown")
        log.info(f"Runner type: {runner_type}")
        
        # Runner-specific configuration display handled by subclasses
        self._display_runner_config(runner_config)
        
        # Ask for confirmation
        log.info("=" * 60)
        confirm = input("Do you want to proceed with the job? [y/N]: ")
        return confirm.lower() in ['y', 'yes']
    
    def _display_runner_config(self, runner_config: Dict[str, Any]) -> None:
        """
        Display runner-specific configuration. To be implemented by subclasses.
        
        Args:
            runner_config: Runner configuration
        """
        pass
    
    def _split_by_groups(
        self,
        runs_df: pd.DataFrame,
        array_mapping: Dict[int, Any]
    ) -> Dict[int, Path]:
        """
        Split the data into separate CSV files for each group.
        
        Args:
            runs_df: DataFrame with run data
            array_mapping: Mapping from array indices to group IDs
            
        Returns:
            Dictionary mapping array indices to CSV file paths
        """
        if not self.script_dir.exists():
            self._prepare_directories()
            
        # Create a data directory for group CSVs
        data_dir = self.script_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        # Dictionary to hold paths to group CSV files
        group_csv_paths = {}
        
        # Process each group
        for array_idx, group_id in array_mapping.items():
            # Filter data for this group
            group_data = runs_df[runs_df[self.grouping_key] == group_id]
            
            # Create CSV file path
            csv_path = data_dir / f"group_{group_id}_{array_idx}.csv"
            
            # Save to CSV
            group_data.to_csv(csv_path, index=False)
            
            # Store the path
            group_csv_paths[array_idx] = csv_path
            
        return group_csv_paths
    
    def process_job_array(self, runs_df: pd.DataFrame) -> List[str]:
        """
        Process and prepare jobs for all groups.
        
        Args:
            runs_df: DataFrame with run data
            
        Returns:
            List of job identifiers
        """
        self._prepare_directories()
        
        # Create mapping from array indices to group IDs
        array_mapping = self._create_array_mapping(runs_df)

        # Save the array mapping to a CSV file
        mapping_file = self.script_dir / "array_mapping.csv"
        mapping_df = pd.DataFrame(
            list(array_mapping.items()), 
            columns=['array_idx', self.grouping_key]
        )
        mapping_df.to_csv(mapping_file, index=False)
        
        # Split the data by groups and get paths to individual CSV files
        group_csv_paths = self._split_by_groups(runs_df, array_mapping)
        
        # Save the paths mapping to a CSV file
        paths_file = self.script_dir / "group_csv_paths.csv"
        paths_data = [[array_idx, str(csv_path)] for array_idx, csv_path in group_csv_paths.items()]
        paths_df = pd.DataFrame(paths_data, columns=['array_idx', 'csv_path'])
        paths_df.to_csv(paths_file, index=False)
        
        # Generate commands for each group using the command_generator
        group_commands = self.command_generator(runs_df, array_mapping, self.grouping_key)
        
        # Execute commands according to the runner-specific implementation
        return self._execute_commands(group_commands, array_mapping)
    
    def _execute_commands(self, group_commands: Dict[int, List[Any]], array_mapping: Dict[int, Any]) -> List[str]:
        """
        Execute commands according to the runner-specific implementation.
        
        Args:
            group_commands: Dictionary mapping array indices to command lists
            array_mapping: Dictionary mapping array indices to group IDs
                
        Returns:
            List of job identifiers
        """
        raise NotImplementedError("Subclasses must implement this method")


class SlurmForeachRunRunner(BaseForeachRunRunner):
    """
    Runner for processing groups of runs using SLURM job arrays.
    """
    
    def __init__(
        self, 
        runner: SlurmRunner, 
        command_generator: CommandGeneratorType, 
        grouping_key: str = None, 
        experiment_dir: Union[str, Path] = Path("experiments")
    ):
        """
        Initialize the SLURM runner.
        
        Args:
            runner: SlurmRunner instance
            command_generator: Function that generates commands for each group
            grouping_key: The column to group runs by (default: "run_id")
            experiment_dir: Base directory for experiments
        """
        super().__init__(command_generator, grouping_key, experiment_dir)
        self.runner = runner
    
    def _display_runner_config(self, runner_config: Dict[str, Any]) -> None:
        """
        Display SLURM-specific configuration.
        
        Args:
            runner_config: Runner configuration
        """
        if runner_config.get("runner") == "slurm":
            log.info("SLURM configuration:")
            for key, value in runner_config.items():
                if key != "runner" and value is not None:
                    log.info(f"  {key}: {value}")
            
            if runner_config.get("job_array_max_concurrency"):
                log.info(f"  Max concurrent array jobs: {runner_config['job_array_max_concurrency']}")
    
    def _execute_commands(self, group_commands: Dict[int, List[Any]], array_mapping: Dict[int, Any]) -> List[str]:
        """
        Execute commands using SLURM job arrays.
        
        Args:
            group_commands: Dictionary mapping array indices to command lists
            array_mapping: Dictionary mapping array indices to group IDs
                
        Returns:
            List of submitted job IDs
        """
        # Get array size
        array_size = len(array_mapping)
        
        script_content = [
            "# Get the array task ID",
            "TASK_ID=$SLURM_ARRAY_TASK_ID",
            "",
            "# Select the command based on the task ID",
            "case $TASK_ID in"
        ]
        
        # Add a case for each array index
        for array_idx, group_id in array_mapping.items():
            # Get the command for this group
            cmd = group_commands.get(array_idx, ["echo", f"No command for group {group_id}"])
            cmd_str = " ".join([str(c) for c in cmd])
            script_content.append(f"    {array_idx})")
            script_content.append(f"        {cmd_str}")
            script_content.append("        ;;")
        
        # Add default case and close the case statement
        script_content.append("    *)")
        script_content.append('        echo "Invalid array task ID: $TASK_ID"')
        script_content.append("        exit 1")
        script_content.append("        ;;")
        script_content.append("esac")
        
        # The command for the SLURM job is to execute this script
        job_id = f"fe-{self.grouping_key}"

        # Submit the job
        try:
            log.info(f"Submitting job array with {array_size} tasks")
            job_ids = self.runner.run_script(
                script=script_content,
                job_id=job_id,
                array_size=array_size,
                array_parallelism=self.runner.config.array_parallelism,
                script_dir=self.script_dir,
                log_dir=self.log_dir,
                env=os.environ.copy()
            )
            log.info(f"Submitted job array: {', '.join(job_ids)}")
            return job_ids
        except Exception as e:
            log.error(f"Failed to submit job array: {e}")
            raise


class SubprocessForeachRunRunner(BaseForeachRunRunner):
    """
    Runner for processing groups of runs using local subprocesses attached to the current terminal.
    """
    
    def _display_runner_config(self, runner_config: Dict[str, Any]) -> None:
        """
        Display subprocess-specific configuration.
        
        Args:
            runner_config: Runner configuration
        """
        if runner_config.get("runner") == "subprocess":
            log.info("Subprocess configuration:")
            for key, value in runner_config.items():
                if key != "runner" and value is not None:
                    log.info(f"  {key}: {value}")
    
    def _execute_commands(self, group_commands: Dict[int, List[Any]], array_mapping: Dict[int, Any]) -> List[str]:
        """
        Execute commands using local subprocesses attached to the current terminal.
        
        Args:
            group_commands: Dictionary mapping array indices to command lists
            array_mapping: Dictionary mapping array indices to group IDs
                
        Returns:
            List of process IDs
        """
        process_ids = []
        
        # Run each command as a subprocess
        for array_idx, group_id in array_mapping.items():
            cmd = group_commands.get(array_idx, ["echo", f"No command for group {group_id}"])
            
            # Convert all arguments to strings to prevent TypeError
            cmd = [str(arg) for arg in cmd]
            
            log.info(f"Running command for group {group_id} (index {array_idx})")
            log.info(f"Command: {' '.join(cmd)}")
            
            try:
                # Run the command as a subprocess attached to the current terminal
                process = subprocess.Popen(
                    cmd, 
                    stdout=None,  # Use parent's stdout (attached to terminal)
                    stderr=None,   # Use parent's stderr (attached to terminal)
                )
                process_ids.append(str(process.pid))
                
                # Wait for the process to complete
                process.wait()
                
                exit_code = process.returncode
                if exit_code == 0:
                    log.info(f"Completed command for group {group_id} successfully")
                else:
                    log.warning(f"Command for group {group_id} exited with code {exit_code}")
                    
            except Exception as e:
                log.error(f"Failed to run command for group {group_id}: {e}")
                raise
        
        return process_ids