import os
import logging
import subprocess

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


class SlurmConfig(BaseModel):
    """Configuration for SLURM job submission with all optional fields"""

    partition: Optional[str] = Field(default=None)
    nodes: Optional[int] = Field(default=None, ge=1)  # must be >= 1 if specified
    ntasks_per_node: Optional[int] = Field(default=None, ge=1)
    cpus_per_task: Optional[int] = Field(default=None, ge=1)
    gres: Optional[str] = Field(default=None)
    time: Optional[str] = Field(default=None)  # in the slurm time format
    mem_gb: Optional[int] = Field(default=None, ge=1)  # memory must be positive
    job_name: Optional[str] = Field(default=None)
    log_dir: Optional[Path] = Field(default=None)
    wckey: Optional[str] = Field(default=None)
    array_size: Optional[int] = Field(default=None, ge=1)
    array_parallelism: Optional[int] = Field(default=None, ge=1)

    @field_validator("gres")
    @classmethod
    def validate_gres(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            parts = v.split(":")
            if len(parts) < 2 or not parts[-1].isdigit():
                raise ValueError(
                    'gres must be in format "resource:number" or "resource:specifier...:number, e.g., "gpu:1"'
                )
        return v

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]):
        """
        Parse a dictionary from YAML into a SlurmAgentConfig object.
        Converts the log_dir string to Path if present.
        """
        # Convert log_dir to Path if it exists
        if config_dict.get("log_dir"):
            config_dict["log_dir"] = Path(config_dict["log_dir"])

        # Remove None values to let SlurmAgentConfig use its defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return SlurmConfig(**config_dict)

    @staticmethod
    def load_slurm_config(file_path: str):
        """
        Load and parse a YAML configuration file into a SlurmAgentConfig object.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            SlurmAgentConfig object
        """
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return SlurmConfig.from_dict(config_dict)


class SlurmRunner:
    """Generic runner for SLURM jobs"""
    
    def __init__(
        self,
        config: Optional[SlurmConfig] = None,
        additional_slurm_params: Optional[Dict[str, Union[str, int, float]]] = None,
    ):
        self.config = config or SlurmConfig()
        self.additional_params = additional_slurm_params or {}
    
    def _generate_sbatch_parameters(
        self,
        job_name: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ) -> List[str]:
        """Generate the SLURM sbatch parameters based on configuration"""
        sbatch_params = []
        
        # Add parameters from configuration
        if self.config.partition is not None:
            sbatch_params.append(f"--partition={self.config.partition}")
        if self.config.nodes is not None:
            sbatch_params.append(f"--nodes={self.config.nodes}")
        if self.config.ntasks_per_node is not None:
            sbatch_params.append(f"--ntasks-per-node={self.config.ntasks_per_node}")
        if self.config.cpus_per_task is not None:
            sbatch_params.append(f"--cpus-per-task={self.config.cpus_per_task}")
        if self.config.gres is not None:
            sbatch_params.append(f"--gres={self.config.gres}")
        if self.config.time is not None:
            sbatch_params.append(f"--time={self.config.time}")
        if self.config.mem_gb is not None:
            sbatch_params.append(f"--mem={self.config.mem_gb}GB")
        if self.config.wckey is not None:
            sbatch_params.append(f"--wckey={self.config.wckey}")
        
        # Use provided job name or fall back to config
        effective_job_name = job_name or self.config.job_name
        if effective_job_name is not None:
            sbatch_params.append(f"--job-name={effective_job_name}")
        
        # Use provided log directory or fall back to config
        effective_log_dir = log_dir or self.config.log_dir
        if effective_log_dir is not None:
            sbatch_params.extend([
                f"--output={effective_log_dir}/%x-%j.out",
                f"--error={effective_log_dir}/%x-%j.err",
            ])
        
        # Add any additional parameters
        for key, value in self.additional_params.items():
            sbatch_params.append(f"--{key}={value}")
        
        return sbatch_params
    
    def generate_sbatch_script(
        self,
        body: str,
        job_name: Optional[str] = None,
        log_dir: Optional[Path] = None,
        env_setup_commands: Optional[List[str]] = None,
    ) -> str:
        """
        Generate the content of a SLURM sbatch script
        
        Args:
            command: The command to run in the SLURM job
            job_name: Optional name for the job (overrides config)
            log_dir: Optional directory for log files (overrides config)
            env_setup_commands: Optional list of commands for environment setup
            
        Returns:
            String content of the sbatch script
        """
        # Generate SBATCH parameters
        sbatch_params = self._generate_sbatch_parameters(job_name, log_dir)
        
        # Generate script content
        script_lines = ["#!/bin/bash", "# SLURM Parameters"]
        
        # Add SBATCH parameters
        script_lines.extend([f"#SBATCH {param}" for param in sbatch_params])
        
        # Add environment setup if provided
        if env_setup_commands:
            script_lines.extend(["", "# Environment setup"])
            script_lines.extend(env_setup_commands)
        
        # Add the command
        script_lines.extend(["", "", body if isinstance(body, str) else "\n".join(body)])
        
        return "\n".join(script_lines)
    
    def submit_job(
        self,
        script_path: Path,
        env: Optional[Dict[str, str]] = None,
        array_size: Optional[int] = None,
        array_parallelism: Optional[int] = None,
    ) -> List[str]:
        """
        Submit a job to SLURM using a script file
        
        Args:
            script_path: Path to the sbatch script
            env: Optional environment variables to pass to the sbatch command
            array_size: Optional size for job array
            array_parallelism: Optional parallelism for job array
            
        Returns:
            List of job IDs
        """
        
        # Add array parameter if needed
        sbatch_cmd = ["sbatch"]
        if array_size is not None and array_size > 0:
            maybe_parallelism = f'%{array_parallelism}' if array_parallelism else ''
            sbatch_cmd.append(f"--array=0-{array_size-1}{maybe_parallelism}")
        
        sbatch_cmd.append(str(script_path))
        
        # Submit job
        try:
            result = subprocess.run(
                sbatch_cmd,
                env=env or {},
                check=True,
                text=True,
                capture_output=True,
            )
            
            # Extract job ID from sbatch output
            job_id = result.stdout.strip().split()[-1]
            
            log.info(f"Submitted job {job_id}")
            return [job_id]
            
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to submit job: {e}")
            log.error(f"sbatch output: {e.output}")
            raise
    
    def run_script(
        self,
        script: Union[str, List[str]],
        job_id: str,
        script_dir: Path,
        log_dir: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        array_size: Optional[int] = None,
        array_parallelism: Optional[int] = None,
        env_setup_commands: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Run a command on SLURM (generates script and submits)
        
        Args:
            command: The command to run in the SLURM job
            job_id: Unique identifier for this job (used in file naming)
            script_dir: Directory to store the script
            log_dir: Optional directory for log files
            env: Optional environment variables to pass to the job
            array_size: Optional size for job array
            array_parallelism: Optional parallelism for job array
            env_setup_commands: Optional list of commands for environment setup
            
        Returns:
            List of job IDs
        """
        # Ensure directories exist
        script_dir.mkdir(parents=True, exist_ok=True)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"Logs will be saved to {log_dir}")
        
        # Generate script content
        script_content = self.generate_sbatch_script(
            body=script,
            job_name=job_id,
            log_dir=log_dir,
            env_setup_commands=env_setup_commands,
        )
        
        # Write script to file
        script_path = script_dir / f"{job_id}.sbatch"
        script_path.write_text(script_content)
        log.debug(f"Launcher script written to {script_path}")
        log.debug(f"Script content:\n{script_content}")
        
        # Make script executable
        script_path.chmod(0o755)
        
        # Submit job
        return self.submit_job(
            script_path=script_path,
            env=env,
            array_size=array_size,
            array_parallelism=array_parallelism,
        )
    
    def check_status(self):
        """Check status of SLURM jobs for current user"""
        squeue_output_format = "%.11i %.12j %.8u %.5D %.2t %4C %7m %16b %.10M %.12P %26R"
        
        squeue_command = [
            "squeue",
            "-u", os.environ.get("USER"),
            "-o", squeue_output_format,
        ]
        
        try:
            result = subprocess.run(
                squeue_command,
                capture_output=True,
                text=True,
                check=True,
            )
            
            log.info(f"Your running SLURM Jobs:\n{result.stdout}")
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to run squeue: {e}")
            log.error(f"squeue output: {e.stderr}")
            raise