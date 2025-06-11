import pandas as pd

from .conftest import python


def test_eval(cifar10_squeezenet1_0_path, temp_root_dir):
    eval_dir = temp_root_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    runs_and_models_csv = eval_dir / "models.csv"

    pd.DataFrame([("sample_model", str(cifar10_squeezenet1_0_path.resolve()))]).to_csv(
        runs_and_models_csv, index=False, header=False
    )

    stdout, _ = python(
        f"-u -m protopnet eval {runs_and_models_csv.absolute()} --output-file={eval_dir}/metrics.pkl --dataset=CIFAR10 --acc-only",
        {"WANDB_MODE": "dryrun"},
    )
    assert f"Loading model from {cifar10_squeezenet1_0_path.absolute()}" in stdout
    assert "Eval complete for sample_model" in stdout
    assert f"Writing Metrics to {eval_dir.resolve()}" in stdout
