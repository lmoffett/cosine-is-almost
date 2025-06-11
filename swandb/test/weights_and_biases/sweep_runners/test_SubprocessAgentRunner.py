import signal
import subprocess
from unittest.mock import Mock, call, patch

import pytest

from swandb.weights_and_biases.sweep_runners import SubprocessAgentRunner


@pytest.fixture
def runner():
    return SubprocessAgentRunner()


def test_get_child_processes_success(runner):
    """Test successful retrieval of child processes."""
    mock_output = "123\n456\n789\n"
    with patch("subprocess.check_output", return_value=mock_output):
        children = runner.get_child_processes(100)
        assert children == [123, 456, 789]
        subprocess.check_output.assert_called_once_with(
            ["ps", "--no-headers", "-o", "pid", "--ppid", "100"], text=True
        )


def test_get_child_processes_no_children(runner):
    """Test when process has no children."""
    with patch("subprocess.check_output", return_value=""):
        children = runner.get_child_processes(100)
        assert children == []


def test_get_child_processes_error(runner):
    """Test handling of subprocess.CalledProcessError."""
    with patch(
        "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "cmd")
    ):
        children = runner.get_child_processes(100)
        assert children == []


def test_terminate_process_tree(runner):
    """Test process tree termination."""
    mock_children = [101, 102]

    with patch(
        "swandb.weights_and_biases.sweep_runners.SubprocessAgentRunner.get_child_processes",
        return_value=mock_children,
    ), patch("os.kill") as mock_kill:
        runner.terminate_process_tree(100)

        # Verify kills were called in correct order (children first, then parent)
        mock_kill.assert_has_calls(
            [
                call(101, signal.SIGTERM),
                call(102, signal.SIGTERM),
                call(100, signal.SIGTERM),
            ]
        )


def test_terminate_nonexistent_process(runner):
    """Test handling of nonexistent process."""
    with patch(
        "swandb.weights_and_biases.sweep_runners.SubprocessAgentRunner.get_child_processes",
        return_value=[101],
    ), patch("os.kill", side_effect=ProcessLookupError):
        # Should not raise an exception
        runner.terminate_process_tree(100)


def test_run_wandb_agent_normal_completion(runner):
    """Test normal completion of wandb agent."""
    mock_proc = Mock()
    mock_proc.pid = 100
    mock_proc.wait = Mock()

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen, patch(
        "os.environ.copy", return_value={"WANDB_SERVICE": "test"}
    ):
        runner.run_wandb_agents("sweep-123")

        # Verify Popen was called correctly
        mock_popen.assert_called_once()
        popen_args = mock_popen.call_args[0][0]
        assert popen_args == ["wandb", "agent", "sweep-123"]

        # Verify environment handling
        env = mock_popen.call_args[1]["env"]
        assert "WANDB_SERVICE" not in env

        # Verify process was waited on
        mock_proc.wait.assert_called_once()


def test_run_wandb_agent_keyboard_interrupt(monkeypatch, runner):
    """Test handling of KeyboardInterrupt during wandb agent execution."""
    mock_proc = Mock()
    mock_proc.pid = 100
    mock_proc.wait = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setenv("WANDB_SERVICE", "test")

    with patch("subprocess.Popen", return_value=mock_proc), patch(
        "swandb.weights_and_biases.sweep_runners.SubprocessAgentRunner.terminate_process_tree"
    ) as mock_terminate, patch("sys.exit") as mock_exit:
        runner.run_wandb_agents("sweep-123")

        # Verify process tree was terminated
        mock_terminate.assert_called_once_with(100, sig=signal.SIGINT)

        # Verify sys.exit was called
        mock_exit.assert_called_once_with(0)


def test_sigint_handler(monkeypatch, runner):
    """Test the SIGINT signal handler."""
    mock_proc = Mock()
    mock_proc.pid = 100
    monkeypatch.setenv("WANDB_SERVICE", "test")

    with patch("subprocess.Popen", return_value=mock_proc), patch(
        "swandb.weights_and_biases.sweep_runners.SubprocessAgentRunner.terminate_process_tree"
    ) as mock_terminate, patch("sys.exit") as mock_exit, patch(
        "signal.signal"
    ) as mock_signal:
        # Start the wandb agent (this will register the signal handler)
        runner.run_wandb_agents("sweep-123")

        # Get the signal handler that was registered
        handler = mock_signal.call_args[0][1]

        # Simulate SIGINT
        handler(signal.SIGINT, None)

        # Verify process tree was terminated
        mock_terminate.assert_called_once_with(100, sig=signal.SIGINT)

        # Verify sys.exit was called
        mock_exit.assert_called_once_with(0)