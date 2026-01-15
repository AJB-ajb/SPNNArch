# SPNNArch: Superposition in Neural Network Architectures

This repository contains the code implementation and thesis for **"Mathematical Models of Superposition in Neural Network Architectures"** (Alexander Busch, December 2025).

## Thesis

The thesis is included in [`thesis.pdf`](./thesis.pdf).

## Repository Structure

### Core Python Library (`src/masterthesis/`)

The python package contains utilities for analyzing the toy model from Elhage et al.:

- **`config.py`** — Configuration management for experiments
- **`experiment_logger.py`** — Experiment logging and result tracking
- **`feature_generation.py`** — Feature generation utilities for toy models
- **`icfg.py`** — Instantiable configuration system for models
- **`plotting_utils.py`** — Visualization and plotting utilities
- **`stacked_torch_modules.py`** — PyTorch modules for stacked architectures
- **`toy_models_metrics.py`** — Metrics for analyzing toy models of superposition
- **`trainer.py`** — Training utilities and loops
- **`utils.py`** — General utility functions

### Python Experiments (`src/masterthesis/experiments/`)

Experiment scripts implementing the analyses from the thesis:

- **`asymptotic_feature_representation.py`** — Asymptotic analysis of feature representations
- **`feat_repr_over_training.py`** — Feature representation evolution during training
- **`feature_WtW_visualization.py`** — Weight matrix visualization
- **`feature_geometry_stability.py`** — Stability analysis of feature geometry
- **`feature_metrics_comparison.py`** — Comparison of different feature metrics
- **`activation_function_experiments/`** — Experiments with different activation functions
- **`optimizer_comparison/`** — Optimizer performance comparisons
- **`scaling_experiments/`** — Scaling behavior investigations

### Julia Experiments (`experimentsjl/`)

Julia experiment scripts for specialized analyses:

- **`error_estimates.jl`** — Error estimation analysis
- **`error_estimates_trained.jl`** — Error estimates on trained models
- **`polygon_defs.jl`** — Polygon solution definitions
- **`polygon_solution_full_nb.jl`** — Full polygon solution analysis


## Installation

### Python Environment

This project uses Poetry for Python dependency management:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Julia Environment

To set up the Julia environment:

```bash
# Navigate to Julia package directory
cd JlCode

# Activate the project and install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Running Tests

### Python Tests

Run the Python test suite using pytest:

```bash
poetry run pytest test/ -v
```

### Julia Tests

Run the Julia test suite:

```bash
julia --project=JlCode JlCode/test/runtests.jl
```

## Usage Examples

### Python: Analyzing Superposition

```python
from masterthesis.experiment_logger import ExpLogger
from masterthesis.feature_generation import generate_features
from masterthesis.toy_models_metrics import compute_superposition_metrics

# Set up experiment
logger = ExpLogger("superposition_analysis")

# Generate features and analyze
features = generate_features(n_features=100, n_dims=50)
metrics = compute_superposition_metrics(features)

logger.log_metrics(metrics)
```

### Julia: Compositional Analysis

```julia
using JlCode

# Load compositional superposition models
include("JlCode/src/CompSuperpos.jl")

# Run analysis
results = analyze_compositional_superposition(params)
```

## Library Components

### Instantiable Configuration (ICfg)

The `icfg.py` module provides a configuration system for serializing and instantiating PyTorch models with their hyperparameters, enabling reproducible experiments.

### Experiment Logger

Both Python and Julia implementations provide experiment logging for tracking:
- Model configurations
- Training metrics
- Visualizations
- Data artifacts

### Stacked Modules

The `stacked_torch_modules.py` implements architectures for studying superposition in stacked linear models with shared weights across instances.

## Reproducing Thesis Results

The experiments in the thesis can be reproduced by running the scripts in `src/masterthesis/experiments/` and `experimentsjl/`. 

## License

- **Code**: [MIT License](./LICENSE) — Free to use, modify, and distribute
- **Thesis**: [CC BY 4.0](./LICENSE-THESIS) — Free to share and adapt with attribution

**Keywords:** Neural Networks, Superposition, Feature Learning, Interpretability, Mathematical Modeling, Deep Learning Theory
