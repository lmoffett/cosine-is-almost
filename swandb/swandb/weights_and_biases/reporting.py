import logging

import click
import pandas as pd
import wandb
from tqdm import tqdm

from ..cli import EnvDefaultOption

log = logging.getLogger(__name__)

def wandb_options(f):
    f = click.option(
        "--wandb-entity", cls=EnvDefaultOption, envvar="WANDB_ENTITY", required=True
    )(f)
    f = click.option(
        "--wandb-project", cls=EnvDefaultOption, envvar="WANDB_PROJECT", required=True
    )(f)
    return f

@click.group()
def reporting():
    """CLI for interacting with WandB runs and exporting data."""
    pass

@reporting.command('runs')
@wandb_options
@click.option(
    "--output", "-o", default="wandb-runs.csv", help="Output CSV file path"
)
@click.option(
    "--internal-metadata", is_flag=True, default=False, help="Include internal metadata in the report"
)
def export_runs(wandb_entity, wandb_project, output, internal_metadata):
    """Export WandB runs data to CSV."""
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    project_path = f"{wandb_entity}/{wandb_project}"
    log.info(f"Fetching runs from {project_path}...")
    
    runs = api.runs(project_path)

    rows = []
    for run in tqdm(runs):
        # .summary contains the output keys/values for metrics like accuracy.
        # We call ._json_dict to omit large files
        summary = run.summary._json_dict

        # .config contains the hyperparameters.
        # We remove special values that start with _.
        config = {k: v for k, v in run.config.items() if not k.startswith('_') or internal_metadata}

        maybe_sweep_id = {}
        if run.sweep:
            maybe_sweep_id = {"sweep_id": run.sweep.id}

        rows.append({
            "name": run.name,
            **config,
            **summary,
            **maybe_sweep_id
        })

    if not rows:
        log.warning("No runs found.")
        return

    runs_df = pd.DataFrame.from_records(rows)
    runs_df.to_csv(output, index=False)
    log.info(f"Exported {len(rows)} runs to {output}")


@reporting.command("sweeps")
@wandb_options
@click.option(
    "--output", "-o", default="wandb-sweeps.csv", help="Output CSV file path"
)
def export_sweeps(wandb_entity, wandb_project, output):
    """Export WandB sweeps metadata to CSV."""
    api = wandb.Api()
    log.info(f"Fetching sweeps from {wandb_entity}/{wandb_project}...")
    
    project = api.project(name=wandb_project, entity=wandb_entity)

    sweeps = project.sweeps()
    
    rows = []
    for sweep in tqdm(sweeps):
        sweep_info = {
            "id": sweep.id,
            "name": sweep.name,
        }
        
        # Extract sweep configuration
        if hasattr(sweep, "config") and sweep.config:
            for key, value in sweep.config.items():
                # Skip internal keys
                if not key.startswith('_'):
                    # Prefix with 'config_' to avoid name collisions
                    sweep_info[f"config_{key}"] = value
        
        # Get summary metrics across all runs in the sweep
        if hasattr(sweep, "best_run"):
            try:
                best_run = sweep.best_run()
                if best_run:
                    sweep_info["best_run_id"] = best_run.id
                    sweep_info["best_run_name"] = best_run.name
            except Exception as e:
                log.warning(f"Warning: Could not fetch best run for sweep {sweep.id}: {e}")
        
        rows.append(sweep_info)

    if not rows:
        log.warning("No sweeps found.")
        return

    sweeps_df = pd.DataFrame.from_records(rows)
    sweeps_df.to_csv(output, index=False)
    log.info(f"Exported {len(rows)} sweeps to {output}")