import csv
import datetime
import logging
from pathlib import Path
from typing import Dict, Optional

import click
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import datasets
from .utilities.trainer_utilities import is_single_valued_metric

from .train.metrics import InterpretableTrainingMetrics

log = logging.getLogger(__name__)


def format_metrics(metrics):
    return "|".join(
        f"{k}:{v:.3g}" for k, v in metrics.items() if type(v) is float or type(v) is int
    )


def run_simple_eval_epoch(model, dataloader, num_classes, device, acc_only=False):
    """
    Run a simple evaluation loop.

    Args:
    - model: The model you want to evaluation. It assumes model has prototype_layer.num_prototypes_per_class.
    - dataloader: The evaluation dataloader.
    - num_classes: The total number of classes in the evaluation dataloader, to initiate InterpretableTrainingMetrics.

    Returns:
    - Dataframe of calculated metrics, {metric name: metric value}.
    """

    interp_metrics = InterpretableTrainingMetrics(
        num_classes=num_classes,
        # FIXME - this assumes that the model has a prototype_layer
        proto_per_class=model.prototype_layer.num_prototypes_per_class if not acc_only else 1,
        # FIXME: these shouldn't be hardcoded
        # instead, the train_dataloaders should return an object
        part_num=15,
        img_size=224,
        half_size=36,
        protopnet=model,
        device=device,
        acc_only=acc_only,
    )

    model.eval()
    interp_metrics.start_epoch('eval')
    with torch.no_grad():
        for _, batch_data_dict in tqdm(enumerate(dataloader)):
            batch_data_dict["img"] = batch_data_dict["img"].to(device)
            batch_data_dict["target"] = batch_data_dict["target"].to(device)
            output = model(
                batch_data_dict["img"], return_prototype_layer_output_dict=True
            )
            interp_metrics.update_all(batch_data_dict, output, phase="project")

        epoch_metrics_dict = interp_metrics.compute_dict()

        for k, v in epoch_metrics_dict.items():
            # FIXME - update this when the interface for the metrics dict is more clearly defined.
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 0:
                    epoch_metrics_dict[k] = v.item()
                else:
                    epoch_metrics_dict[k] = v.cpu().numpy()
            else:
                out = []
                for vv in v:
                    if len(vv) == 1:
                        out.append(vv.item())
                    elif isinstance(vv, torch.Tensor):
                        out.append(vv.cpu().numpy())
                    elif isinstance(vv[0], torch.Tensor):
                        out.append([vvi.cpu().numpy() for vvi in vv])
                    else:
                        out.append(np.array(vv))

                epoch_metrics_dict[k] = tuple(out)

    del interp_metrics
    return epoch_metrics_dict


def load_and_eval(
    models: Dict[str, str],
    val_dataloader,
    test_dataloader,
    device,
    num_classes,
    acc_only=False,
):
    """
    Run evaluations on the selected runs.

    Args:
    - models - A dictionary of model_name: model_path.
    - dataloader: The evaluation dataloader.
    - device: The device all calculations will be on.

    Returns:
    - Dataframe of calculated metrics.
    """

    fail_to_load = []
    processed_metrics = []

    for i, (model, save_path) in enumerate(models.items()):
        log.info("Loading model from %s", save_path)
        try:
            saved_model = torch.load(save_path, map_location=torch.device(device))
        except Exception as e:
            log.error(f"{e}-{save_path} failed to load")
            fail_to_load.append(save_path)
            processed_metrics.append({"model": model, "model_path": save_path, "error": str(e)})
            continue

        try:

            log.info("Running val metrics for %s", model)
            val_metrics = run_simple_eval_epoch(
                saved_model,
                val_dataloader,
                num_classes=num_classes,
                device=device,
                acc_only=acc_only,
            )

            log.info("Running test metrics for %s", model)
            test_metrics = run_simple_eval_epoch(
                saved_model,
                test_dataloader,
                num_classes=num_classes,
                device=device,
                acc_only=acc_only,
            )

            complexity = saved_model.get_prototype_complexity()
            model_metrics = {
                "n_unique_protos": complexity["n_unique_proto_parts"],
                "n_unique_proto_parts": complexity["n_unique_proto_parts"],
                "n_protos": saved_model.num_prototypes,
            }

            log.info("Eval complete for %s, (%s/%s)", model, i + 1, len(models))

            val_metrics = {f"val.{k}": v for k, v in val_metrics.items() if is_single_valued_metric(v)}
            test_metrics = {f"test.{k}": v for k, v in test_metrics.items() if is_single_valued_metric(v)}
            model_metrics = {f"model.{k}": v for k, v in model_metrics.items()}

            metrics = {
                "model": model,
                "model_path": save_path,
                **val_metrics,
                **test_metrics,
                **model_metrics,
            }

            log.info("%s Metrics Summary: %s", model, format_metrics(metrics))

            processed_metrics.append(metrics)
        
        except Exception as e:
            log.error(f"Error processing model {model}: {e}")
            fail_to_load.append(save_path)
            processed_metrics.append({"model": model, "model_path": save_path, "error": str(e)})
        finally:
            del saved_model

    metric_df = pd.DataFrame(processed_metrics).set_index("model")

    return metric_df


class ModelParameter(click.ParamType):
    name = "model"

    def convert(self, value, param, ctx):
        if not value:
            return None

        value = str(value)

        # Check if it's a PyTorch model file
        if value.endswith((".pth", ".pt", ".ckpt", ".model", ".bin")):
            return {"model": value}

        # Check if it's a CSV file with model mappings
        elif value.endswith(".csv"):
            try:
                return self.parse_model_csv(value)
            except Exception as e:
                self.fail(f"Failed to parse CSV file: {e}", param, ctx)

        else:
            self.fail(
                "Unsupported file type. Please provide a PyTorch model file (.pth, .pt, .ckpt, .model, .bin) or a CSV file (.csv)",
                param,
                ctx,
            )

    def parse_model_csv(self, file_path: str) -> Dict[str, str]:
        """Parse a CSV file containing model identifier and path pairs."""
        pairs = {}
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2 and not row[0].startswith("#"):
                    identifier, path = row[0].strip(), row[1].strip()
                    if identifier and path:
                        pairs[identifier] = path
        return pairs


@click.command("eval")
@click.argument(
    "model",
    type=ModelParameter(),
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    help="Path to save the output csv file",
)
@click.option(
    "--device",
    type=str,
    default=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    help="Device to run the evaluation on",
)
@click.option("--batch-size", type=int, default=100, help="Batch size for evaluation")
@click.option(
    "--dataset",
    type=click.Choice(
        ["CUB200", "CUB200_CROPPED", "CARS", "CARS_CROPPED", "DOGS", "CIFAR10"],
        case_sensitive=False,
    ),
    default="CUB200",
    help="Dataset to evaluate on",
)
@click.option("--acc-only", is_flag=True, help="Only calculate accuracy", default=False)
def run(
    model: Dict[str, str],
    output_file: Optional[Path],
    device: str,
    batch_size: int,
    dataset: str,
    acc_only: bool,
):
    """
    Evaluate models listed in a CSV file.

    models: Path to either a PyTorch model file (.pth, .pt, .ckpt, .model, .bin) or a CSV file with model identifier-path pairs
    output_dir: Path to save the output csv file.
    device: Device to run the evaluation on.
    batch_size: Batch size for evaluation.
    dataset: Dataset to evaluate on.
    acc_only: Only calculate accuracy.
    """

    # Convert device string to torch.device
    device = torch.device(device)

    batch_sizes = {"train": 1, "project": 1, "val": batch_size, "test": batch_size}

    split_dataloaders = datasets.training_dataloaders(dataset, batch_sizes=batch_sizes)

    num_classes = split_dataloaders.num_classes
    val_loader = split_dataloaders.val_loader
    test_loader = split_dataloaders.test_loader

    log.info("Starting evaluation on %s models: %s", len(model), model.keys())

    result_df = load_and_eval(
        model,
        val_loader,
        test_loader,
        device=device,
        num_classes=num_classes,
        acc_only=acc_only,
    )

    if output_file is None:
        output_file = (Path.cwd() / ("eval" + "-".join(model.keys()))).with_suffix(
            ".pkl"
        )

    output_filename = output_file.stem

    true_output_file = (
        (output_file.parent / f"{output_filename}_{datetime.datetime.now()}")
        .with_suffix(".pkl")
        .resolve()
    )

    log.info("Writing Metrics to %s", true_output_file)
    result_df.to_pickle(true_output_file)

    link = output_file
    log.info(f"Symlinking {link} to {true_output_file}")
    link.unlink(missing_ok=True)
    link.symlink_to(true_output_file.name)
