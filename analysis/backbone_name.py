from pathlib import Path
from typing import Any, Dict, List

import click
import pandas as pd
import torch
from tqdm import tqdm


def command(
    runs_df: pd.DataFrame,
    array_mapping: Dict[int, Any],
    grouping_key: str,
) -> Dict[int, List[str]]:

    commands = {}
    for array_idx, group_id in array_mapping.items():
        group_df = runs_df[runs_df[grouping_key] == group_id]
        commands[array_idx] = [
            "python",
            "analysis/backbone_name.py",
            group_id,
            *group_df["best_model"].tolist(),
        ]
    return commands


def get_activation(model):
    try:
        act = model.prototype_layer.activation_function
    except Exception:
        act = model.models[0].prototype_layer.activation_function
    return str(act).split(" ")[0].split(".")[-1]


def get_num_classes(model):
    try:
        num_classes = model.prototype_layer.num_classes
    except Exception:
        try:
            num_classes = model.models[0].prototype_layer.num_classes
        except Exception:
            num_classes = model.prototype_layer.num_prototypes
    return num_classes


@click.command()
@click.argument("group", type=str)
@click.argument("model_paths", type=click.Path(exists=True), nargs=-1)
def load_model(group: str, model_paths: str):
    """
    group: The group name
    model_paths: The paths to the models
    """
    output_dir = Path("analysis/eval/model-backbones")
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_strs = []
    for model_path in tqdm(model_paths):
        model = torch.load(model_path)
        backbone_strs.append(
            (
                model_path,
                str(type(model)),
                str(model.backbone.embedded_model),
                get_activation(model),
                get_num_classes(model),
            )
        )

    pd.DataFrame(backbone_strs).to_csv(
        output_dir / f"{group}-backbones.csv", index=False, header=False
    )
    print("Backbone names saved to", output_dir / f"{group}-backbones.csv")
