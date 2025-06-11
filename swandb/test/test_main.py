from importlib import import_module

from click.testing import CliRunner
from conftest import skip_if_not_wandb


@skip_if_not_wandb
def test_train_without_sweep():
    runner = CliRunner()
    main = import_module("swandb.__main__")

    # Invoke the command
    result = runner.invoke(
        main.cli,
        [
            "sweep",
            "train",
            "test/mock_train.py::train",
            "param1=3",
            "param2=30",
            "baseline=300",
        ],
        env={"WANDB_OFFLINE": "true"},
    )

    assert result.exit_code == 0
    assert "wandb: Run summary:" in result.output, result.output
    assert "wandb: mock_target_metric 333" in result.output, result.output
