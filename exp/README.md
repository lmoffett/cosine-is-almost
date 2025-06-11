# Experiments

This directory contains all experiment configurations and scripts needed to reproduce the experiment sweeps.

## Experiment Types

### Short Experiments (`short/`)

These correspond to the **Cosine Ablation** experiments (Section 3.1) with limited GPU time:

- 120 GPU hours per sweep  
- Tests cosine vs. Euclidean similarity across model variants
- Uses successive halving for early elimination of poor runs
- Designed for rapid comparison of distance metrics

### Long Experiments (`long/`)  

These correspond to the **Bayesian Optimized Results** (Section 3.2) with full optimization:

- 216 GPU hours per sweep
- More thorough search over hyperparameter space intended to measure the possible performance under cosine

## Directory Structure

```
exp/
├── short/                  # Cosine ablation experiments  
│   ├── *.py                # Training script for model
│   ├── *.yaml              # Sweep configurations
│   └── sbatch.yaml         # SLURM configuration
├── long/                   # Bayesian optimization experiments
│   ├── vanilla_accuracy.py # Full optimization training script
│   ├── vanilla_accproto.py # Full optimization of joint accuracy and interpretability objective
│   ├── *.yaml              # Full optimization sweep configs  
│   └── sbatch.yaml         # SLURM configuration for long runs
└── README.md               # This file
```

## Setup

Ensure your environment is configured (see [root README](../README.md)) and activate the conda environment:

```bash
conda activate exp-cos
export $(grep -v '^#' .env | xargs)
```

## Running Experiments

### 1. Preflight Testing

Before launching full sweeps, run preflight tests to verify your setup:

```bash
# Test short experiment configuration
swandb sweep launch \
    exp/short/vanilla-accuracy.yaml dataset=cub200 \
    --runner-config=exp/short/sbatch.yaml \
    --preflight

# Test long experiment configuration  
swandb sweep launch \
    exp/long/vanilla-accuracy_cub.yaml dataset=cub200 \
    --runner-config=exp/long/sbatch.yaml \
    --preflight
```

Preflight runs use the parameters in `preflight_parameters` sections to quickly test:
- Environment configuration
- Data loading
- Model initialization
- Training pipeline

Check that your preflight sweep runs to completion and reports a target metric for each configuration.
Launching the training prints log directories, which can be used for debugging.
Stack traces are *not* in the weights and biases interface, and are instead in the local .err files.

### 2. Launching Short Experiments (Cosine Ablation)

Run the cosine ablation experiments:

```bash
# Launch vanilla ProtoPNet comparison
swandb sweep launch \
    exp/short/vanilla-accuracy.yaml \
    --runner-config=exp/short/sbatch.yaml

# Launch other model variants
swandb sweep launch exp/short/deformable.yaml --runner-config=exp/short/sbatch.yaml
swandb sweep launch exp/short/prototree.yaml --runner-config=exp/short/sbatch.yaml
```

*Note*: This will launch a sweep for distance metric, dataset, and backbone combinations.
In general, you want to select them independently, as was done in the paper:

```bash
# Launch vanilla ProtoPNet comparison
swandb sweep launch \
    exp/short/vanilla-accuracy.yaml \
    --runner-config=exp/short/sbatch.yaml
    dataset=cub200 backbone=resnet50[pretraining=inaturalist] backbone=densenet161 backbone=vgg19 activation_function=l2
```

### 3. Launching Long Experiments (Bayesian Optimization)

Run full optimization experiments one dataset at a time:

```bash
# CUB-200 optimization
swandb sweep launch \
    exp/long/vanilla-accuracy.yaml dataset=cub200 \
    --runner-config=exp/long/sbatch.yaml

# Other datasets  
swandb sweep launch exp/long/vanilla-accuracy.yaml dataset=cub200_cropped --runner-config=exp/long/sbatch.yaml
swandb sweep launch exp/long/vanilla-accuracy.yaml dataset=cars_cropped --runner-config=exp/long/sbatch.yaml
swandb sweep launch exp/long/vanilla-accuracy.yaml dataset=dogs --runner-config=exp/long/sbatch.yaml
```

The same caveat about launching individual sweeps applies as above.

### 4. Auto-Launch Mode

After initial verification, use auto-launch to skip interactive confirmation:

```bash
swandb sweep launch exp/long/vanilla-accuracy.yaml dataset=cub200 \
    --runner-config=exp/long/sbatch.yaml --auto-launch
```

## Configuration Files

### Sweep Configurations (.yaml)

Each experiment is defined by a YAML file containing:

```yaml
description: "Experiment description"
method: bayes                    # Optimization method  
metric:
  goal: maximize
  name: best[prototypes_embedded]/eval/accuracy

parameters:                      # Hyperparameters to optimize
  learning_rate:
    distribution: log_normal
    mu: -3
    sigma: 1
  
hyper_sweep_parameters:          # Experimental factors
  backbone:
    values: [resnet50, densenet161, vgg19]
  activation_function:  
    values: [cosine, l2]
  dataset:
    values: [cub200, cars_cropped, dogs]

preflight_parameters:            # Fixed values for testing
  learning_rate: 0.001
  batch_size: 32
```

### SLURM Configurations (sbatch.yaml)

```yaml
runner: slurm
cpus_per_task: 8
gres: gpu:a5000:1               # Adjust GPU type as needed
time: 3-12:00:00                # Walltime limit
mem_gb: 32
array_size: 3                   # Number of parallel jobs
```

## Compute Environment Adaptation

### SLURM Clusters

The default configuration assumes SLURM with A5000 GPUs.
Modify `sbatch.yaml` files for your cluster:

```yaml
gres: gpu:v100:1                # Change GPU type
mem_gb: 64                      # Adjust memory  
time: 1-00:00:00               # Reduce time for testing
```

### Local Execution

For local testing or non-SLURM environments:

```bash
swandb sweep launch exp/short/vanilla-accuracy.yaml --runner-config subprocess
```

This runs only a single agent process and therefore has limited parallelism.

### Array Size Adjustment

Control parallelism with `--runner-config-set`:

```bash
swandb sweep launch exp/long/vanilla-accuracy.yaml \
    --runner-config=exp/long/sbatch.yaml \
    --runner-config-set array_size 12
```

This is commen in preflight testing where you may want to test many configurations for a short time at once.

## Expected Outputs

### During Training

- **W&B logs**: Real-time metrics at `wandb.ai/{entity}/{project}`
- **Checkpoints**: Temporary files in `$PPNXT_CHECKPOINT_DIR`  
- **SLURM logs**: In experiment directory with job IDs

### After Completion

- **Final models**: Saved to `$PPNXT_ARTIFACT_DIR`
- **Results data**: Exportable via `swandb reporting` commands
- **Metrics**: Best scores logged to W&B run summaries

## Monitoring Progress

### W&B Dashboard

Monitor experiments at: `https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}`

Key metrics to watch:

- `best[prototypes_embedded]/eval/accuracy` - Final model accuracy
- `best[prototypes_embedded]/eval/prototype_sparsity` - Model sparsity
- Training phase transitions and early stopping

`[prototypes_embedded]` refers to the prototypes having been projected and the backbone parameters **not** changing.
This is critical to maintaining interpretability and is the actual optimization objective.


## Troubleshooting

### Common Issues

**CUDA out of memory**:

- Reduce batch sizes in sweep configuration
- Use gradient accumulation: modify `num_accumulation_batches`

**SLURM job failures**:

- Verify GPU resources are available
- Review job logs in experiment directories

**W&B authentication**:

```bash
wandb login
# Enter your API key from wandb.ai/settings
```