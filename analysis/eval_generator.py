"""
Example command generator module.

This can be referenced from the CLI using:
path/to/example_command_generator.py::echo
"""

import os
import pathlib
from typing import Any, Dict, List

import pandas as pd


def eval(
    runs_df: pd.DataFrame,
    array_mapping: Dict[int, Any],
    grouping_key: str,
) -> Dict[int, List[str]]:
    """
    Example command generator that creates a Python command for each group.

    Args:
        runs_df: The full merged DataFrame
        array_mapping: Dict mapping array indices to group IDs
        grouping_key: The column used for grouping

    Returns:
        Dict mapping array indices to lists of command arguments
    """
    commands = {}
    data_dir = pathlib.Path(os.environ["SWANDB_EVAL"]) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for array_idx, group_id in array_mapping.items():
        # Get rows for this group
        group_df = runs_df[runs_df[grouping_key] == group_id]

        datasets = group_df["dataset"].unique()
        assert len(datasets) == 1, "Each group should have exactly one dataset"
        dataset = datasets[0]

        backbones = group_df["backbone"].unique()
        assert len(backbones) == 1, "Each group should have exactly one backbone"
        backbone = backbones[0]

        if backbone == "dense161":
            maybe_batch_flags = ["--batch-size", "32"]
        else:
            maybe_batch_flags = []

        group_file = data_dir / f"{group_id}-models.csv"

        group_df[["run_id", "best_model"]].to_csv(group_file, index=False, header=False)

        maybe_acc_only = (
            ["--acc-only"] if os.environ.get("ACC_ONLY", "").lower() == "true" else []
        )

        commands[array_idx] = (
            [
                "python",
                "-m",
                "protopnet",
                "eval",
                str(group_file),
                "--dataset",
                dataset,
                "--output-file",
                str(data_dir / f"{group_id}-results.csv"),
            ]
            + maybe_acc_only
            + maybe_batch_flags
        )
    return commands
