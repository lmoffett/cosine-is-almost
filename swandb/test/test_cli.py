import click
import pytest
from click import testing

from swandb.cli import EnvDefaultOption, GenericKeyValueArgs


# Create a dummy command to test the GenericKeyValueArgs
@click.command()
@click.argument("kwargs", nargs=-1, type=GenericKeyValueArgs())
def kv_command(kwargs):
    click.echo(dict(kwargs))


@pytest.fixture
def runner():
    """Fixture for click CLI runner."""
    return testing.CliRunner()


def test_valid_key_value_pairs(runner):
    """Test valid key=value pairs."""
    result = runner.invoke(kv_command, ["key1=value1", "key2=value2"])
    assert result.exit_code == 0
    assert result.output.strip() == "{'key1': 'value1', 'key2': 'value2'}"


def test_missing_equals_sign(runner):
    """Test invalid input missing '='."""
    result = runner.invoke(kv_command, ["key1value1"])
    assert result.exit_code != 0
    assert "Invalid argument format" in result.output


def test_empty_key_value_pairs(runner):
    """Test empty input."""
    result = runner.invoke(kv_command, [])
    assert result.exit_code == 0
    assert result.output.strip() == "{}"


def test_partial_key_value_pair(runner):
    """Test key with empty value."""
    result = runner.invoke(kv_command, ["key1="])
    assert result.exit_code == 0
    assert result.output.strip() == "{'key1': ''}"


def test_multiple_equals_signs(runner):
    """Test value with multiple '=' signs."""
    result = runner.invoke(kv_command, ["key1=value=1"])
    assert result.exit_code == 0
    assert result.output.strip() == "{'key1': 'value=1'}"


@click.command()
@click.option(
    "--api-key",
    cls=EnvDefaultOption,
    envvar="API_KEY",
    help="API key for the service.",
)
def env_command(api_key):
    click.echo(f"API key: {api_key}")


def test_flag_provided(runner):
    """Test when the flag is provided directly."""
    result = runner.invoke(env_command, ["--api-key", "12345"])
    assert result.exit_code == 0
    assert result.output.strip() == "API key: 12345"


def test_env_var_provided(monkeypatch, runner):
    """Test when the environment variable is set."""
    monkeypatch.setenv("API_KEY", "env_api_key")
    result = runner.invoke(env_command, [])
    assert result.exit_code == 0
    assert result.output.strip() == "API key: env_api_key"


def test_flag_overrides_env_var(monkeypatch, runner):
    """Test when both the flag and environment variable are set."""
    monkeypatch.setenv("API_KEY", "env_api_key")

    result = runner.invoke(env_command, ["--api-key", "flag_api_key"])
    assert result.exit_code == 0
    assert result.output.strip() == "API key: flag_api_key"


def test_missing_flag_and_env_var(runner):
    """Test when neither the flag nor the environment variable is set."""
    result = runner.invoke(env_command, [])
    assert result.exit_code != 0
    assert (
        "One of the flag or environment variable (API_KEY) must be set."
        in result.output
    )
