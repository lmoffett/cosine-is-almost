# Cosine Similarity is *Almost* All You Need (for Prototypical-Part Models)

This repository contains the code and experiments for the workshop paper **Cosine Similarity is *Almost* All You Need (for Prototypical-Part Models)**, CVPR xAI4CV Workshop, 2025.

## Repository Structure

```
├── analysis/         # Code for analyzing experimental results
├── exp/              # Experiment configurations and training scripts
├── protopnext/       # Core ProtoPNet library
├── swandb/           # CLI tool for managing Weights & Biases sweeps
├── user_study/       # Code for user study data sampling and processing
├── .env              # Environment variable template
└── conda-env.yaml    # Conda environment specification
```

### Directory Overview

- **`analysis/`** - Contains scripts for analyzing sweep results, extracting model statistics, and generating evaluation metrics
- **`exp/`** - Experiment definitions, training scripts, and SLURM configurations for running the paper's experiments  
- **`protopnext/`** - The core ProtoPNet library with implementations of various prototype-based models
- **`swandb/`** - A CLI wrapper around Weights & Biases for managing hyperparameter sweeps on SLURM clusters (originally created for SLURM automation; non-SLURM clusters require custom runner implementation)
- **`user_study/`** - Scripts for processing and preparing data for the human evaluation study

## Quick Start

### 1. Environment Setup

Create a python 3.8 environment, e.g. with conda:
```bash
conda env create -f conda-env.yaml
conda activate exp-cos
```

Install the local packages:
```bash
pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu117
```

### VSCode Setup (Recommended)

This repository is configured for development in VSCode. For the best experience:

1. **Open as Multi-root Workspace**: Add `protopnext/` and `swandb/` as separate root folders in your VSCode workspace to enable debugging and testing those submodules directly.

2. **Workspace Configuration**: 
   ```json
   {
     "folders": [
       {"path": "."},
       {"path": "./protopnext"}, 
       {"path": "./swandb"}
     ]
   }
   ```

3. **Python Interpreter**: Set the Python interpreter to your conda environment (`exp-cos`) for each root folder.

### 2. Configure Environment Variables

Environment variables control the local file paths for all datasets, training data, and analysis.
A template is in `.env` and descriptions of each environment variable are below.

vscode automatically loads the environment variables in .env into your workspace, but if you run directly from a terminal, you must set those variables, which you can do with:

```{bash}
export $(grep -v '^#' .env | xargs)
```

### 3. Dataset Setup

Download datasets to the directories specified in your `.env`:

- **CUB-200-2011**: [Caltech-UCSD Birds](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- **Stanford Cars**: [Cars Dataset](https://www.kaggle.com/datasets/hassiahk/stanford-cars-dataset-full)  (note the original dataset is no longer available, but a copy can be obtained from the Kaggle at the link)
- **Stanford Dogs**: [Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

**Prepare datasets** using the three-stage process:

```bash
# 1. Download data (manual step to directories above)

# 2. Prep metadata (which is specific to dataset format
python -m protopnet datasets prep-metadata $CUB200_DIR --dataset cub200-cropped
python -m protopnet datasets prep-metadata $CARS_DIR --dataset cars-cropped  
python -m protopnet datasets prep-metadata $DOGS_DIR --dataset dogs

# 3. Create train/val splits (per dataset)
python -m protopnet datasets create-splits $CUB200_DIR --image-dir images_cropped
```

### 4. Run Experiments

See the [experiment README](exp/README.md) for detailed instructions on replicating the paper's results.

#### Nomenclature

- **`short`** - Refers to the **Cosine Ablation** experiments from the paper (Section 3.1)
- **`long`** - Refers to the **Bayesian Optimized Results** experiments (Section 3.2)

## Environment Variables

The following environment variables must be configured in your `.env.local` file:

### Dataset Paths

```bash
CUB200_DIR=/path/to/CUB_200_2011           # CUB-200-2011 dataset
CARS_DIR=/path/to/stanford_cars             # Stanford Cars dataset  
DOGS_DIR=/path/to/stanford_dogs             # Stanford Dogs dataset
```

### Experiment Paths

```bash
PPNXT_ARTIFACT_DIR=/path/to/artifacts       # Where models and results are saved
PPNXT_CHECKPOINT_DIR=/path/to/checkpoints   # Temporary checkpoints during training
PPNXT_MODEL_DIR=/path/to/pretrained         # Pretrained model cache
TORCH_HOME=/path/to/torch_cache             # PyTorch model cache
XDG_CACHE_HOME=/path/to/cache               # General cache directory
```

### Weights & Biases Configuration

```bash
WANDB_DIR=/path/to/wandb                    # W&B logging directory
WANDB_ENTITY=your_wandb_entity              # Your W&B username/team
WANDB_PROJECT=your_project_name             # W&B project name
```

### SWANDB Configuration  
```bash
SWANDB_EVAL=/path/to/eval                   # Evaluation output directory
SWANDB_LOG_DIR=/path/to/logs                # Training logs
SWANDB_EXPERIMENT_BASE_DIR=/path/to/exp     # Base experiment directory
```

### User Study Configuration
```bash
STUDY_DIR=/path/to/study                    # User study data directory
STUDY_MODEL_PROTO_OUT=/path/to/proto_out    # Prototype output directory
STUDY_CSV_TRIPLET_PATH=user_study/attention-samples.csv  # Sample data file
STUDY_CSV_OUTPUT_DIR=/path/to/study_output  # Processed study output
```

## Reproducing Paper Results

**Paper Experiment Mapping**:

- Table 1 / Section 3.1: `exp/short/{model}.yaml` (5 GPU x 24 hours), where {model} is one of ProtoPNet, Deformable, ProtoTree, TesNet, or ST-ProtoPNet
- Figure 2 / Section 3.2: `exp/long/vanilla-accuracy.yaml` (3 GPU x 72 Hours)

**Reproduction Notes**: Set `PPNXT_SEED=1234`, but note that CUDA deterministic was not enabled.

## Results Data

The complete run metadata and statistics are available in `analysis/wandb_runsXsweepsXeval.csv`. This file is denormalized and contains all experiments:

- Model and experiment metadata
- All hyperparameter configurations tested
- Final accuracy and sparsity metrics  
- Training metadata and convergence information

If you replicate the experiments, you can regenerate this dataset with the `swandb reporting` commands (see [swandb README](swandb/README.md)) and the notebooks in `analysis`.

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (experiments run on A5000, debugging on P100)
- **RAM**: 32GB+ recommended  
- **Storage**: ~100GB for datasets and results
- **Expected Runtimes**: Short experiments (24 hours), Long experiments (3 days)

## Known Issues

This revision is the code used in the actual experiments, and had the following bugs that did not impact the results:

### Validation Metrics Rounding Bug

There is a known bug where validation metrics are rounded during training but not during evaluation. This affects the display of metrics in W&B logs but does not impact model selection or final results.
The bug does not occur during standalone evaluation.

### Early Stopping does not Move Checkpoints

When early stopping is triggered, the checkpoints are not moved to the artifact directory.
The checkpoints are still usable as-is, but this causes critical model weights to be in multiple places on the filesystem.

### Naming Collisions

The name generation entropy lead to naming collisions when running thousands of training runs.
The entropy was increased midway through the experiments, but handling previous runs required cross-checking results in `analysis/ExportResults.ipynb`.

## Non-Archival Workshop Paper

This work was presented as a non-archival workshop paper.
If you use this code or build upon these ideas, please consider citing the arXiv version of the ProtoPNeXt paper for the underlying ProtoPNeXt library [7].

## Contributors

- Frank Willard
- Maximilian Machado
- Emanuel Mokel
- Adam Costarino
- Jon Donnelly
- Zhicheng Guo
- Dennis Tang
- Julia Yang
- Giyoung Kim
- Alina Jade Barnett

## References

1. Chen et al. This Looks Like That: Deep Learning for Interpretable Image Recognition. NeurIPS, 2019.
2. Donnelly et al. Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes. ICCV, 2022.
3. Nauta et al. Neural prototype trees for interpretable fine-grained image recognition. ICCV, 2021.
4. Wang et al. Interpretable Image Recognition by Constructing Transparent Embedding Space. ICCV, 2021.
5. Wang et al. Learning support and trivial prototypes for interpretable image classification. ICCV, 2023
6. Huang et al. Evaluation and improvement of interpretability for self-explainable part-prototype networks. ICCV, 2023.
7. Willard, Frank, et al. "This looks better than that: Better interpretable models with protopnext." arXiv preprint arXiv:2406.14675 (2024).

## License

[MIT License](./LICENSE.txt)
