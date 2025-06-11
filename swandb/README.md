# swandb

`swandb` (SLURM for Weights & Biases) is a tool for managing Weights & Biases (W&B) hyperparameter sweeps in SLURM environments.

## Features

- Run W&B sweeps on SLURM clusters or locally
- Pre-validate sweep configurations with preflight mode
- Support for both Python method-based and command-based training approaches
- Flexible parameter management with support for regular and hyper-sweep parameters

## Background

SWANDB was originally created to automate Weights & Biases hyperparameter sweeps on SLURM clusters. While it includes a basic subprocess runner for local execution, **automating sweeps on non-SLURM clusters requires implementing a custom runner**. See the `slurm/` directory for examples of how runners are structured.

## Usage

### Basic Sweep Configuration

Create a YAML configuration file for your sweep:

```yaml
description: "My W&B Sweep"
method: "grid"  # or "random", "bayes"
metric:
  goal: maximize
  name: accuracy

# Regular parameters
parameters:
  learning_rate:
    values: [0.001, 0.01]
  batch_size:
    values: [32, 64]

# High-level parameters for sweep organization
hyper_sweep_parameters:
  model_type:
    values: ["resnet50", "densenet121"]
  dataset:
    values: ["imagenet", "cifar10"]

# Values for preflight validation
preflight_parameters:
  learning_rate: 0.001
  batch_size: 32
```

### Running Sweeps

You can run sweeps using either a Python method or a command:

#### Method-based Approach

```python
# train.py
def train(learning_rate: float, batch_size: int):
    # Your training code here
    pass
```

```bash
# Launch the sweep
swandb sweep launch sweep_config.yaml --wandb-entity your-entity --wandb-project your-project
```

#### Command-based Approach

```yaml
# In your sweep config
program: train.py
command:
  - ${env}
  - python
  - ${program}
  - ${args}
```

### Preflight Validation

Before running a full sweep, validate your configuration:

```bash
swandb sweep launch sweep_config.yaml --preflight
```

This creates a grid-search sweep over every combination in the `hyper_sweep_parameters` where other variables are set to the fixed values in `preflight_parameters`.
Determining `preflight_parameters` is the job of the author - they should exercise all steps in your training regime to ensure they work and you have enough memory to support them.
Moreover, you should do enough training to be sure that it is working.
Beyond that, the goal is to be as computationally efficient as possible.

### SLURM Configuration

For SLURM environments, create a runner configuration file. See [example configuration](test/mock_runner_config.yaml).

```bash
swandb sweep launch sweep_config.yaml --runner-config /path/to/runner_config.yaml
```

### Subprocess Runner

For testing or local development, use the subprocess runner:

```bash
swandb sweep launch sweep_config.yaml --runner-config subprocess
```

This will launch the sweep in a subprocess, but will not release the terminal.

### Additional Options

- logging level: `--log-level {DEBUG,INFO,WARNING,ERROR}`
- seed: `--seed 1234`

## Development Setup

1. Create conda environment:

```bash
conda env create -f conda-env.yaml
```

2. Install development dependencies:

```bash
pip install -e ".[dev]" --extra-index-url=https://download.pytorch.org/whl/cu117
```

3. Set up git hooks:

```bash
pre-commit install
```

## Environment Variables

- `WANDB_ENTITY` - from Weights & Biases
- `WANDB_PROJECT` - from Weights & Biases.
- `SWANDB_SWEEP_RUNNER_CONFIG` - default configuration file for running a sweep. Can be a yaml file or the name of a runner (i.e., `slurm` or `subprocess`). Default is `subprocess`.
- `SWANDB_ARTIFACT_DIR` - Directory on local filesystem to save artifacts and checkpoints.
- `SWANDB_LOG_DIR` - Log directory.

## Reporting and Data Export

SWANDB provides commands to export experiment results from Weights & Biases for analysis.

### Export Sweep Metadata

Export information about all sweeps in your project:

```bash
swandb reporting sweeps --wandb-entity your-entity --wandb-project your-project --output sweeps.csv
```

This exports:

- Sweep configurations and hyperparameters
- Best run information for each sweep
- Sweep metadata and status

### Export Run Results  

Export detailed results from all runs after you have executed your sweeps:

```bash
swandb reporting runs --wandb-entity your-entity --wandb-project your-project --output runs.csv
```

Options:
- `--internal-metadata`: Include internal W&B metadata fields (default: excluded)
- `--output FILE`: Specify output CSV file path

This exports:
- All hyperparameter configurations
- Final metrics and best scores
- Run metadata (duration, status, etc.)
- Model paths and checkpoints