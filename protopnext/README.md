# ProtoPNext

**This is a prerelease codebase. All APIs, interfaces, and implementations are subject to change without notice.**

ProtoPNext is a research library for prototypical part networks (ProtoPNets) that provides implementations of various prototype-based interpretable models including ProtoPNet, TesNet, Deformable ProtoPNet, ProtoTree, and ST-ProtoPNet.

**Note**: This library can be used independently for model development and experimentation.
The experiment framework (`exp/` directory) and swandb are only needed for large-scale hyperparameter sweeps - you can work directly with the models using the CLI or Python API.

## Code Organization

```
protopnet/
├── models/              # Model implementations
│   ├── vanilla_protopnet.py
│   ├── deformable_protopnet.py
│   └── ...
├── datasets/            # Dataset loaders and utilities
│   ├── torch_extensions.py
│   ├── cars_cropped.py
│   └── ...
├── train/              # Auto-training infrastructure
│   ├── scheduling/     # Training phase scheduling
│   ├── logging/        # Metrics and W&B integration
│   └── checkpointing.py
├── utilities/          # General utilities
│   ├── trainer_utilities.py
│   ├── visualization_utilities.py
│   └── ...
├── pretrained/         # Pretrained backbone implementations
├── activations.py      # Prototype activation functions
├── backbones.py        # Backbone model construction
├── embedding.py        # Feature embedding layers
├── prediction_heads.py # Classification heads
├── prototype_layers.py # Prototype computation layers
└── cli.py              # Command-line interface
```

### Core Components vs Training Infrastructure

The codebase is organized into two main categories:

**Core Library Components** are used for constructing models:

- `activations.py` - Cosine, L2, and other prototype activation functions
- `backbones.py` - Backbone model construction and configuration
- `embedding.py` - Feature embedding and addon layers  
- `prediction_heads.py` - Classification heads for prototype models
- `prototype_layers.py` - Core prototype computation and projection
- `models/` - Complete model implementations
- `datasets/` - Dataset loading and preprocessing
- `pretrained/` - Pretrained model implementations
- `utilities/` - General utilities for visualization, preprocessing, etc.

**Auto-Training Infrastructure** (`train/` package) is used for creating training loops:

- `scheduling/` - Multi-phase training schedules and early stopping
- `logging/` - Metrics tracking, W&B integration, and logging
- `checkpointing.py` - Model saving and loading utilities
- `types.py` - Training-related protocols and interfaces

## Key Concepts

### Prototype Models

All models inherit from `ProtoPNet` base class and consist of:

- **Backbone**: Feature extraction (ResNet, DenseNet, VGG, etc.)
- **Add-on layers**: Feature processing before prototype computation
- **Prototype layer**: Computes similarity between features and learned prototypes
- **Prediction head**: Maps prototype activations to class predictions

### Activation Functions

- **Cosine similarity**: Angular distance between features and prototypes
- **L2 distance**: Euclidean distance (traditional ProtoPNet)
- **Exp L2**: Exponential of negative L2 distance

### Training Phases

Training alternates between multiple phases:

- **Warm-up**: Train only prototype layers (backbone frozen)
- **Joint**: End-to-end training of all components
- **Project**: Update prototypes to match training examples
- **Last-only**: Train only the classification head

### Early Stopping

Custom early stopping for prototype models:

- **Project patience**: Stop after N projections without improvement

## Command Line Interface

The library provides a CLI to train models based on command line configuration, which can be used as a starting point for experimenting with ProtoPNet models.
Custom models will require their own configurations.

```bash
# Train vanilla ProtoPNet
python -m protopnet train-protopnet --dataset cub200 --backbone resnet50

# Train deformable ProtoPNet  
python -m protopnet train-deformable --dataset cub200 --activation-function cosine

# Evaluate models
python -m protopnet eval models.csv --dataset cub200 --output results.csv

# Generate visualizations
python -m protopnet visualization model.pth --dataset cub200
```

## Datasets

Available datasets:

- **cub200**: CUB-200-2011 birds dataset
- **cub200_cropped**: Cropped version using bounding boxes
- **cars_cropped**: Stanford Cars with cropping
- **dogs**: Stanford Dogs dataset

## Library 

### Training Phase System

Training alternates between *phases*, which are a series of *steps*.
`ClassificationBackpropPhase` is the standard gradient descent training phase, but configureswhich network layers to train.
Other phases have special non-gradient behaviors, like projection.

For example, the `ProtoPNetTrainingSchedule` has the following layout:

- **Warm-up**: `ClassificationBackpropPhase` training only prototype layers (and, optionally, last layer)
- **Joint**: `ClassificationBackpropPhase` training all layers
- **Project**: `ProjectPhase` - one epoch updating the prototypes to match the closest sample patches
- **Last-only**: `ClassificationBackpropPhase` training only the classification head

## Known Issues


### VSCode Configuration

This package is designed to work as a separate root folder in VSCode multi-root workspaces. Add `protopnext/` as a root folder to enable:
- Proper Python path resolution
- IntelliSense and autocomplete
- Integrated debugging and testing

### Code Style

The project uses:

- Black for code formatting
- isort for import sorting  
- flake8 for linting
- pre-commit hooks for automated checking

### Testing

Run unit tests with:
```bash
pytest protopnet/test
```

Run integration tests with:
```bash
pytest protopnet/inttest
```

## Dependencies

Core dependencies:

- PyTorch >= 1.13.1
- torchvision >= 0.14.1
- wandb >= 0.16.6
- pandas >= 2.0.2
- matplotlib >= 3.6.3
- opencv-python >= 4.7.0

See `pyproject.toml` for complete dependency list.
