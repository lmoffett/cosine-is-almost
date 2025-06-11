import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


from ..slurm.slurm_runner import SlurmConfig, SlurmRunner

log = logging.getLogger(__name__)


class SubprocessAgentRunner:
    def get_child_processes(self, pid):
        """
        Get the child processes of a given parent process ID (pid) using `ps`.
        """
        try:
            # Use the ps command to find child processes of the given pid
            result = subprocess.check_output(
                ["ps", "--no-headers", "-o", "pid", "--ppid", str(pid)],
                text=True,
            )
            # Parse the result into a list of child PIDs
            return [int(line.strip()) for line in result.splitlines() if line.strip()]
        except subprocess.CalledProcessError:
            # If `ps` fails (e.g., if the process doesn't exist anymore), return an empty list
            return []

    def terminate_process_tree(self, pid, sig=signal.SIGTERM):
        """
        Terminate a process and all its children using `ps` to query child processes.
        """
        try:
            # Get the child processes of the parent process
            children = self.get_child_processes(pid)
            # Kill child processes first
            for child_pid in children:
                log.info(f"Terminating child process {child_pid}")
                os.kill(child_pid, sig)

            # Then, kill the parent process
            log.info(f"Terminating parent process {pid}")
            os.kill(pid, sig)
        except ProcessLookupError:
            log.warning(f"Process {pid} does not exist, skipping.")

    def run_wandb_agents(self, sweep_id):
        # Start the wandb agent subprocess
        env = os.environ.copy()
        # the subprocess should get its own service
        if "WANDB_SERVICE" in env:
            del env["WANDB_SERVICE"]
        proc = subprocess.Popen(
            ["wandb", "agent", sweep_id],
            preexec_fn=os.setpgrp,  # Start in a new process group
            env=env,  # Pass the current environment variables
        )

        def handle_sigint(sig, frame):
            """
            Handle Ctrl+C (SIGINT) and terminate the wandb agent and its children.
            """
            log.info("Received Ctrl+C. Terminating wandb agent and its subprocesses...")
            self.terminate_process_tree(proc.pid, sig=sig)  # Terminate the process tree
            sys.exit(0)  # Exit the parent process

        # Register the signal handler for SIGINT
        signal.signal(signal.SIGINT, handle_sigint)

        try:
            # Wait for the wandb agent process to complete
            proc.wait()
            log.info("wandb agent process completed.")
        except KeyboardInterrupt:
            # Handle Ctrl+C during the wait
            handle_sigint(signal.SIGINT, None)

    def check_runner_status(self):
        pass


class SlurmWandBAgentRunner:
    """Runs W&B agents on SLURM (uses SlurmRunner under the hood)"""
    
    def __init__(
        self,
        slurm_runner: Optional[SlurmRunner] = None,
        slurm_config: Optional[SlurmConfig] = None,
        experiment_dir: Optional[Path] = None,
    ):
        if slurm_runner and slurm_config:
            raise ValueError("Only one of slurm_runner or slurm_config should be provided.")
        self.slurm_runner = slurm_runner or SlurmRunner(config=slurm_config)
        self.experiment_dir = experiment_dir or Path.cwd() / "experiments"
    
    def run_wandb_agents(
        self, 
        sweep_id: str, 
        env: Optional[Dict[str, str]] = os.environ,
        array_size: Optional[int] = None,
    ) -> List[str]:
        """
        Run W&B agents on SLURM
        
        Args:
            sweep_id: W&B sweep ID
            env: environment variables. Normally, you will set this to os.environ.
            array_size: Optional size for job array
            
        Returns:
            List of job IDs
        """
        # Create directory structure for this sweep
        sweep_dir = self.experiment_dir / "sweeps" / sweep_id
        log_dir = sweep_dir / "logs"
        artifacts_dir = sweep_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup W&B-specific environment
        wandb_env = os.environ.copy()
        if env is not None:
            wandb_env.update(env)
        if "WANDB_SERVICE" in wandb_env:
            del wandb_env["WANDB_SERVICE"]

        wandb_env["SWANDB_ARTIFACT_DIR"] = str(artifacts_dir)
        wandb_env["SWANDB_LOG_DIR"] = str(log_dir)
        
        # Build W&B command
        command = f"wandb agent {sweep_id}"
        
        # Use the generic SlurmRunner to run the command
        return self.slurm_runner.run_script(
            script=command,
            job_id=f"wba-{sweep_id}",
            script_dir=sweep_dir,
            log_dir=log_dir,
            env=wandb_env,
            array_size=array_size,
        )
    
    def check_runner_status(self):
        """
        Check the status of the SLURM runner.
        """
        return self.slurm_runner.check_status()