# Project Agents and Workflow

This document outlines the automated agents (scripts) that constitute the MLOps pipeline for this project. These agents are designed to be run sequentially to handle hyperparameter optimization, model training, cross-validation, and results analysis in a structured and reproducible manner.

## The ML Workflow

The project follows a standard yet robust machine learning workflow, orchestrated by a series of command-line scripts. The typical order of operations is as follows:

1.  **Hyperparameter Optimization**: Find the best model architecture and training parameters.
2.  **Model Training**: Perform multiple runs with the best parameters to ensure statistical significance. After each run, cross-validation is performed.
3.  **Analysis and Plotting**: The `Analysis` class in `src/Analysis.py` can be used to aggregate the results from training and validation to generate insights.

## Project Configuration and Setup

This project uses `uv` for fast dependency management, as defined in the `pyproject.toml` file.

### Environment Setup

To create a virtual environment and install all necessary packages, run the following commands from the root of the project:

1.  **Create and activate the virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies using `uv`**:
    ```bash
    pip install uv
    uv pip sync requirements.txt
    ```
This ensures you have a reproducible environment with the exact versions of the libraries used for development.

### Coding Style and Conventions

To maintain code quality and readability, this project adheres to the following conventions:

*   **Code Formatting**: All Python code is formatted using the **Black** code formatter to ensure a consistent style.
*   **Type Hinting**: All functions and methods should include type hints for clarity and to enable static analysis.
*   **Docstrings**: All modules, classes, and functions should have clear and concise docstrings explaining their purpose, arguments, and return values.
*   **Modularity**: The project is structured into distinct modules (`src`, `scripts`, `Data`) with clear responsibilities.

---

## Agent 1: Hyperparameter Optimization

There are two specialized agents for this task, depending on the model type.

### `optimize_hyperparameters.py` (for Homogeneous Models)

*   **Purpose**: This agent uses the **Optuna** library to perform an automated search for the best hyperparameters for the homogeneous GNN models (GCN, TAG, Cheb, etc.). It aims to find the set of parameters that minimizes the validation loss.

*   **Inputs**:
    *   A raw training data file (`--train_file`).
    *   The directory containing raw data (`--raw_data_dir`).
    *   A directory to store the Optuna study database (`--storage_dir`).

*   **Outputs**:
    *   An SQLite database file (e.g., `gnn_spn_optimization_gcn.db`) for each GNN operator, containing the results of all trials.

*   **Usage**:
    ```bash
    python scripts/optimize_hyperparameters.py --gnn_operator gcn --n_trials 100 --train_file GridData_DS1_train_data.processed
    ```

### `optimize_hyperparameters_hetero.py` (for Heterogeneous Models)

*   **Purpose**: This agent performs the same function as its homogeneous counterpart but is tailored for heterogeneous models like RGAT and HEAT.

*   **Inputs**:
    *   Raw training and validation data files (`--train_file`, `--val_file`).

*   **Outputs**:
    *   An SQLite database file (e.g., `hetero_gnn_spn_optimization_rgat.db`).

*   **Usage**:
    ```bash
    python scripts/optimize_hyperparameters_hetero.py --gnn_operator rgat --n_trials 50
    ```

---

## Agent 2: Model Training

### `train_model.py`

*   **Purpose**: After identifying the best hyperparameters, this agent performs multiple training runs (`--num_runs`, typically 30) using different random seeds. This rigorous process validates the statistical stability of the model's performance.

*   **Inputs**:
    *   The directory containing the Optuna study databases (`--studies_dir`). The user is prompted to select which optimized model(s) to train.
    *   Training and testing data files.

*   **Outputs**:
    *   A directory for each training run (e.g., `results/state_dicts/gnn_spn_experiment/gcn_run_0_seed_42/`) containing:
        *   `best_model.pt`: The trained model's `state_dict` (weights).
        *   `hparams.json`: The hyperparameters used for that run.
    *   A `statistical_results.parquet` file summarizing the performance metrics from all runs.

*   **Usage**:
    ```bash
    python scripts/train_model.py
    ```
