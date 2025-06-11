from swandb.slurm.slurm_runner import SlurmConfig
from pathlib import Path

import pytest

def test_empty_config():
    config = SlurmConfig()
    assert config.partition is None
    assert config.nodes is None
    assert config.ntasks_per_node is None
    assert config.cpus_per_task is None
    assert config.gres is None
    assert config.time is None
    assert config.mem_gb is None
    assert config.job_name is None
    assert config.log_dir is None
    assert config.wckey is None


def test_full_config():
    config = SlurmConfig(
        partition="gpu",
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=4,
        gres="gpu:1",
        time="00:60:00",
        mem_gb=32,
        job_name="test_job",
        log_dir=Path("/tmp/logs"),
        wckey="test_key",
    )
    assert config.partition == "gpu"
    assert config.nodes == 1
    assert config.ntasks_per_node == 1
    assert config.cpus_per_task == 4
    assert config.gres == "gpu:1"
    assert config.time == "00:60:00"
    assert config.mem_gb == 32
    assert config.job_name == "test_job"
    assert config.log_dir == Path("/tmp/logs")
    assert config.wckey == "test_key"


def test_parse_slurm_config():
    config_dict = {
        "partition": "gpu",
        "nodes": 1,
        "gres": "gpu:1",
        "log_dir": "/tmp/test_logs",
    }
    config = SlurmConfig.from_dict(config_dict)

    assert config.partition == "gpu"
    assert config.nodes == 1
    assert config.gres == "gpu:1"
    assert config.log_dir == Path("/tmp/test_logs")
    assert config.cpus_per_task is None  # Unspecified fields should be None
    assert config.time is None


def test_parse_slurm_config_with_none_values():
    config_dict = {"partition": "gpu", "nodes": None, "gres": "gpu:1", "log_dir": None}
    config = SlurmConfig.from_dict(config_dict)

    assert config.partition == "gpu"
    assert config.nodes is None
    assert config.gres == "gpu:1"
    assert config.log_dir is None


def test_load_slurm_config(temp_dir):
    # Create a temporary YAML file
    config_yaml = """
    partition: gpu
    nodes: 1
    gres: gpu:1
    log_dir: /tmp/test_logs
    """
    config_file = temp_dir / "test_config.yaml"
    config_file.write_text(config_yaml)

    # Load the config
    config = SlurmConfig.load_slurm_config(str(config_file))

    assert config.partition == "gpu"
    assert config.nodes == 1
    assert config.gres == "gpu:1"
    assert config.log_dir == Path("/tmp/test_logs")


def test_load_slurm_config_missing_file():
    with pytest.raises(FileNotFoundError):
        SlurmConfig.load_slurm_config("nonexistent_config.yaml")

