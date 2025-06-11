import subprocess
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from swandb.slurm.slurm_runner import SlurmConfig, SlurmRunner


@pytest.fixture
def slurm_config_minimal():
    """Fixture for minimal SLURM configuration"""
    return SlurmConfig(partition="gpu", gres="gpu:1")


@pytest.fixture
def slurm_config_full(temp_dir):
    """Fixture for full SLURM configuration"""
    return SlurmConfig(
        partition="gpu",
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=4,
        gres="gpu:1",
        time="00:60:00",
        mem_gb=32,
        job_name="test_job",
        log_dir=temp_dir / "logs",
    )

@pytest.fixture
def additional_params():
    """Fixture for additional SLURM parameters"""
    return {
        "constraint": "volta",
        "qos": "normal",
        "account": "my-project",
    }


def test_generate_sbatch_parameters_minimal(slurm_config_minimal, temp_dir):
    """Test SBATCH parameter generation with minimal configuration"""
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Generate parameters
    params = runner._generate_sbatch_parameters(
        job_name="test-job",
        log_dir=temp_dir
    )
    
    # Verify parameters
    assert "--partition=gpu" in params
    assert "--gres=gpu:1" in params
    assert "--job-name=test-job" in params
    assert any(f"--output={temp_dir}" in p for p in params)
    assert any(f"--error={temp_dir}" in p for p in params)
    
    # Verify missing parameters
    assert not any("--nodes=" in p for p in params)
    assert not any("--mem=" in p for p in params)


def test_generate_sbatch_parameters_full(slurm_config_full):
    """Test SBATCH parameter generation with full configuration"""
    runner = SlurmRunner(config=slurm_config_full)
    
    # Generate parameters
    params = runner._generate_sbatch_parameters()
    
    # Verify all parameters are included
    assert "--partition=gpu" in params
    assert "--nodes=1" in params
    assert "--ntasks-per-node=1" in params
    assert "--cpus-per-task=4" in params
    assert "--gres=gpu:1" in params
    assert "--time=00:60:00" in params
    assert "--mem=32GB" in params
    assert "--job-name=test_job" in params
    
    # Verify log paths use the configured directory
    log_dir = slurm_config_full.log_dir
    assert any(str(log_dir) in p for p in params)


def test_generate_sbatch_script_minimal(slurm_config_minimal, temp_dir):
    """Test SBATCH script generation with minimal configuration"""
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Generate script
    command = "echo 'Hello, World!'"
    script_content = runner.generate_sbatch_script(
        body=command,
        job_name="test-job",
        log_dir=temp_dir
    )
    
    # Check script content
    lines = script_content.splitlines()
    assert "#!/bin/bash" in lines
    assert "#SBATCH --partition=gpu" in lines
    assert "#SBATCH --gres=gpu:1" in lines
    assert "#SBATCH --job-name=test-job" in lines
    assert any("--output=" in line for line in lines)
    assert any("--error=" in line for line in lines)
    assert command in lines
    
    # Verify missing parameters
    assert not any("--nodes=" in line for line in lines)
    assert not any("--mem=" in line for line in lines)


def test_generate_sbatch_script_full(slurm_config_full):
    """Test SBATCH script generation with full configuration"""
    runner = SlurmRunner(config=slurm_config_full)
    
    command = "python train.py --epochs 100"
    env_setup = ["module load python", "source /path/to/venv/bin/activate"]
    
    script_content = runner.generate_sbatch_script(
        body=command,
        env_setup_commands=env_setup
    )
    
    lines = script_content.splitlines()
    
    # Verify all parameters are included
    assert "#SBATCH --partition=gpu" in lines
    assert "#SBATCH --nodes=1" in lines
    assert "#SBATCH --ntasks-per-node=1" in lines
    assert "#SBATCH --cpus-per-task=4" in lines
    assert "#SBATCH --gres=gpu:1" in lines
    assert "#SBATCH --time=00:60:00" in lines
    assert "#SBATCH --mem=32GB" in lines
    assert "#SBATCH --job-name=test_job" in lines
    
    # Verify environment setup commands
    assert "module load python" in lines
    assert "source /path/to/venv/bin/activate" in lines
    
    # Verify command
    assert command in lines
    
    # Verify log paths use the configured directory
    log_dir = slurm_config_full.log_dir
    assert any(str(log_dir) in line for line in lines)


def test_generate_sbatch_script_with_additional_params(slurm_config_minimal, additional_params):
    """Test SBATCH script generation with additional parameters"""
    runner = SlurmRunner(
        config=slurm_config_minimal,
        additional_slurm_params=additional_params
    )
    
    script_content = runner.generate_sbatch_script(body="echo 'test'")
    lines = script_content.splitlines()
    
    # Verify additional parameters are included
    assert "#SBATCH --constraint=volta" in lines
    assert "#SBATCH --qos=normal" in lines
    assert "#SBATCH --account=my-project" in lines


@patch("subprocess.run")
def test_submit_job_success(mock_run, temp_dir, slurm_config_minimal):
    """Test successful job submission"""
    # Mock successful sbatch submission
    mock_run.return_value = Mock(stdout="Submitted batch job 12345\n", returncode=0)
    
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Create a simple script file
    script_path = temp_dir / "test_job.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    # Submit the job
    job_ids = runner.submit_job(script_path=script_path)
    
    # Verify sbatch was called correctly
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    
    # Verify sbatch command
    assert args[0][0] == "sbatch"
    assert args[0][1] == str(script_path)
    
    # Verify job ID was returned
    assert job_ids == ["12345"]


@patch("subprocess.run")
@pytest.mark.parametrize("array_paralellism", [None, 2])
def test_submit_job_with_array(mock_run, temp_dir, slurm_config_minimal, array_paralellism):
    """Test job submission with array parameter"""
    mock_run.return_value = Mock(stdout="Submitted batch job 12345\n", returncode=0)
    
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Create a simple script file
    script_path = temp_dir / "test_job.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    # Submit the job as an array
    job_ids = runner.submit_job(script_path=script_path, array_size=4, array_parallelism=array_paralellism)
    
    # Verify sbatch was called correctly
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    
    # Verify array parameter
    assert args[0][0] == "sbatch"
    if array_paralellism is None:
        assert args[0][1] == "--array=0-3"
    else:
        assert args[0][1] == "--array=0-3%2"
    assert args[0][2] == str(script_path)
    
    # Verify job ID was returned
    assert job_ids == ["12345"]


@patch("subprocess.run")
def test_submit_job_with_env(mock_run, temp_dir, slurm_config_minimal):
    """Test job submission with custom environment variables"""
    mock_run.return_value = Mock(stdout="Submitted batch job 12345\n", returncode=0)
    
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Create a simple script file
    script_path = temp_dir / "test_job.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    # Custom environment
    custom_env = {"MY_VAR": "value", "CUDA_VISIBLE_DEVICES": "0"}
    
    # Submit the job with environment
    job_ids = runner.submit_job(script_path=script_path, env=custom_env)
    
    # Verify environment was passed to subprocess.run
    args, kwargs = mock_run.call_args
    assert "env" in kwargs
    assert kwargs["env"]["MY_VAR"] == "value"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"


@patch("subprocess.run")
def test_submit_job_error(mock_run, temp_dir, slurm_config_minimal):
    """Test handling of sbatch submission error"""
    # Mock error in sbatch submission
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, 
        cmd=["sbatch"], 
        output="Error: Invalid partition name specified",
        stderr="Error: Invalid partition name specified"
    )
    
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Create a simple script file
    script_path = temp_dir / "test_job.sh"
    script_path.write_text("#!/bin/bash\necho 'test'")
    
    # Verify error is raised
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        runner.submit_job(script_path=script_path)
    
    assert "Invalid partition name" in str(exc_info.value.stderr)


@patch("subprocess.run")
def test_run_command(mock_run, temp_dir, slurm_config_minimal):
    """Test run_command method which combines script generation and submission"""
    mock_run.return_value = Mock(stdout="Submitted batch job 12345\n", returncode=0)
    
    runner = SlurmRunner(config=slurm_config_minimal)
    
    # Run a simple command
    job_ids = runner.run_script(
        script="echo 'Hello, World!'",
        job_id="test-job",
        script_dir=temp_dir,
        log_dir=temp_dir / "logs"
    )
    
    # Verify sbatch was called
    mock_run.assert_called_once()
    
    # Verify script was created
    script_path = temp_dir / "test-job.sbatch"
    assert script_path.exists()
    
    # Verify script content
    script_content = script_path.read_text()
    assert "echo 'Hello, World!'" in script_content
    
    # Verify script permissions
    assert script_path.stat().st_mode & 0o755 == 0o755
    
    # Verify job ID was returned
    assert job_ids == ["12345"]


@patch("subprocess.run")
def test_check_status(mock_run):
    """Test check_status method"""
    mock_output = """
    JOBID       NAME       USER    NODES  STATE  CPUS  MEMORY       QOS        TIME        PARTITION   NODELIST
    12345       test_job   user1   1      R      4     16GB         normal     0:10:00     gpu         node001
    12346       test_job2  user1   1      PD     4     16GB         normal     0:00:00     gpu         (None)
    """
    mock_run.return_value = Mock(stdout=mock_output, returncode=0)
    
    runner = SlurmRunner()
    output = runner.check_status()
    
    # Verify squeue was called correctly
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    
    # Verify squeue command
    assert args[0][0] == "squeue"
    assert "-u" in args[0]
    
    # Verify output was returned
    assert output == mock_output


def test_slurm_config_validation():
    """Test SlurmConfig validation"""
    # Valid configuration
    config = SlurmConfig(gres="gpu:1")
    assert config.gres == "gpu:1"
    
    # Invalid gres format
    with pytest.raises(ValueError):
        SlurmConfig(gres="gpu")
    
    # Valid complex gres format
    config = SlurmConfig(gres="gpu:volta:2")
    assert config.gres == "gpu:volta:2"


def test_slurm_config_from_dict():
    """Test creating SlurmConfig from dictionary"""
    config_dict = {
        "partition": "gpu",
        "nodes": 1,
        "gres": "gpu:1",
        "log_dir": "/path/to/logs",
        "mem_gb": 16
    }
    
    config = SlurmConfig.from_dict(config_dict)
    
    assert config.partition == "gpu"
    assert config.nodes == 1
    assert config.gres == "gpu:1"
    assert config.mem_gb == 16
    assert isinstance(config.log_dir, Path)
    assert str(config.log_dir) == "/path/to/logs"