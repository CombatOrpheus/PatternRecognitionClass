# GNN for SPN Analysis: Predicting Stochastic Petri Net Properties

This project provides a comprehensive framework for using Graph Neural Networks (GNNs) to analyze and predict key performance metrics of Stochastic Petri Nets (SPNs). It features a full MLOps pipeline, from data processing and hyperparameter optimization to model training, cross-validation, and results analysis.

## About The Project

Stochastic Petri Nets are a powerful formalism for modeling and analyzing concurrent systems. However, traditional analysis methods can be computationally expensive. This project explores the use of GNNs to learn the underlying structure of SPNs and predict their properties, such as average token counts and transition firing rates, offering a potentially faster alternative to simulation.

The repository includes implementations for various homogeneous and heterogeneous GNN architectures, including GCN, TAG, RGAT, and HEAT, all built on PyTorch and PyTorch Lightning.

### Key Features

*   **End-to-End MLOps Pipeline**: Scripts for hyperparameter optimization, training, and analysis.
*   **Flexible Configuration**: Centralized TOML configuration with command-line overrides.
*   **Multiple GNN Architectures**: Supports both homogeneous and heterogeneous models.
*   **Comprehensive Analysis**: Includes tools for statistical tests and generating publication-quality plots.
*   **Reproducibility**: Uses `uv` for dependency management and seeds for reproducible results.

## Getting Started

Follow these instructions to set up the project locally.

### Prerequisites

*   Python 3.10+
*   An NVIDIA GPU is recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    This project uses `uv` for fast dependency management.
    ```bash
    # Install uv if you don't have it
    pip install uv

    # Create and activate the virtual environment
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This command ensures your environment has the exact package versions specified in `pyproject.toml`.
    ```bash
    uv pip sync pyproject.toml
    ```

## MLOps Workflow

The project is structured around a clear and sequential MLOps workflow. The main entry point `scripts/main.py` orchestrates this entire process.

1.  **Hyperparameter Optimization (`scripts/optimize_hyperparameters.py`)**:
    Uses Optuna to search for the best hyperparameters for each GNN model specified in the config. The results are saved in an SQLite database.

2.  **Model Training (`scripts/train_model.py`)**:
    Takes the best parameters from the optimization studies and performs multiple training runs with different random seeds to ensure statistical stability. It saves model checkpoints and statistical results.

3.  **Cross-Validation (`scripts/test_model.py`)**:
    Evaluates the trained models against a collection of unseen datasets to test their generalization capabilities.

4.  **Analysis and Plotting (`src/Analysis.py`)**:
    Aggregates the results from training and cross-validation to generate summary tables, statistical tests (e.g., Friedman test), and visualizations like critical difference diagrams and heatmaps.

## Usage

All scripts are configured via a TOML file. The default is `configs/default_config.toml`. You can create a custom config and use the `--config` flag or override parameters directly from the command line.

### Running the Full Pipeline

To run the entire workflow from optimization to analysis:
```bash
python scripts/main.py
```
This is the recommended way to run a full experiment.

### Running Individual Scripts

You can also run each part of the pipeline individually.

**1. Optimize Hyperparameters**
```bash
# For homogeneous models
python scripts/optimize_hyperparameters.py

# For heterogeneous models
python scripts/optimize_hyperparameters_hetero.py
```

**2. Train Models**
This script uses the results from the optimization step.
```bash
python scripts/train_model.py
```

**3. Evaluate Models on Test Datasets**
```bash
python scripts/test_model.py
```

## Configuration

The project is configured using `configs/default_config.toml`. You can find a detailed explanation of all available options in `configs/CONFIG_OPTIONS.md`.

### Overriding Configuration

You can override any setting from the command line by specifying its path in the TOML file.

**Example:**
To change the number of epochs for training:
```bash
python scripts/train_model.py --training.max_epochs 150
```

## Project Structure

```
.
├── Agents.md               # Instructions for AI agents
├── Data/                   # Raw and processed data
├── GNN tests.ipynb         # Jupyter notebook for experimentation
├── LICENSE
├── README.md               # This file
├── configs/                # Configuration files
│   ├── CONFIG_OPTIONS.md
│   └── default_config.toml
├── environment.yml         # Conda environment file
├── pipelines/
│   └── run_pipeline.sh
├── pyproject.toml          # Project metadata and dependencies for uv
├── requirements.txt
├── scripts/                # Main scripts for running the pipeline
│   ├── main.py
│   ├── optimize_hyperparameters.py
│   ├── optimize_hyperparameters_hetero.py
│   ├── test_model.py
│   └── train_model.py
├── src/                    # Source code
│   ├── Analysis.py
│   ├── CrossValidation.py
│   ├── HeterogeneousModels.py
│   ├── HomogeneousModels.py
│   ├── PetriNets.py
│   ├── ReachabilityGraphDataModule.py
│   ├── SPNDataModule.py
│   ├── SPNDatasets.py
│   ├── config_utils.py
│   └── name_utils.py
└── test/
```

## Contributing

Contributions are welcome! Please follow the existing code style.

*   **Code Formatting**: This project uses **Black** for consistent code formatting.
*   **Type Hinting**: All functions should include type hints.
*   **Docstrings**: Use Google-style docstrings for all modules, classes, and functions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
