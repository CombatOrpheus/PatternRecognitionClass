"""This script serves as the main entry point for running the complete MLOps pipeline.

It orchestrates the entire workflow by sequentially executing the key phases
of the experiment:
1.  **Hyperparameter Optimization**: Finds the best model parameters.
2.  **Model Training**: Trains the final models using the best parameters.
3.  **Analysis**: Generates plots and statistical summaries from the results.

The behavior of each phase is controlled by a central configuration file.
"""
import torch

from src.config_utils import load_config
from src.Analysis import Analysis
from scripts.train_model import main as train_main
from scripts.optimize_hyperparameters import main as optimize_main


def main():
    """Orchestrates the entire experiment workflow.

    This function loads the configuration and then runs the hyperparameter
    optimization, model training, and results analysis phases in sequence.
    """
    torch.set_float32_matmul_precision("high")
    config, config_path = load_config()

    print("--- Starting Experiment Workflow ---")

    # 1. Hyperparameter Optimization Phase
    print("\n--- Phase 1: Hyperparameter Optimization ---")
    optimize_main(config, config_path)

    # 2. Training and Cross-Validation Phase
    print("\n--- Phase 2: Model Training and Cross-Validation ---")
    train_main(config)

    # 3. Analysis Phase
    print("\n--- Phase 3: Analysis and Plotting ---")
    analysis = Analysis(config)
    analysis.run()

    print("\n--- Experiment Workflow Completed ---")


if __name__ == "__main__":
    main()
