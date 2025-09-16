"""This script serves as the main entry point for running the complete MLOps pipeline.

It orchestrates the entire workflow by sequentially executing the key phases
of the experiment:
1.  **Hyperparameter Optimization**: Finds the best model parameters.
2.  **Model Training**: Trains the final models using the best parameters.
3.  **Analysis**: Generates plots and statistical summaries from the results.

The behavior of each phase is controlled by a central configuration file.
"""

import argparse

import torch
import torch_geometric

from scripts.optimize_hyperparameters import main as optimize_main
from scripts.train_model import main as train_main
from src.Analysis import Analysis
from src.config_utils import load_config


def main():
    """Orchestrates the entire experiment workflow.

    This function loads the configuration and then runs the hyperparameter
    optimization, model training, and results analysis phases in sequence.
    """
    parser = argparse.ArgumentParser(description="Main script for the MLOps pipeline.")
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="If set, skips the hyperparameter optimization phase.",
    )
    parser.add_argument(
        "--only-analysis",
        action="store_true",
        help="If set, runs only the analysis phase.",
    )
    args, unknown = parser.parse_known_args()

    torch.set_float32_matmul_precision("high")

    torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
    torch.serialization.add_safe_globals([torch_geometric.data.data.DataTensorAttr])
    torch.serialization.add_safe_globals([torch_geometric.data.storage.GlobalStorage])

    config, config_path = load_config(unknown)

    print("--- Starting Experiment Workflow ---")

    if args.only_analysis:
        print("\n--- Running only the Analysis Phase ---")
        analysis = Analysis(config, config_path)
        analysis.run()
        print("\n--- Analysis Phase Completed ---")
        return

    if not args.skip_optimization:
        # 1. Hyperparameter Optimization Phase
        print("\n--- Phase 1: Hyperparameter Optimization ---")
        optimize_main(config, config_path)
    else:
        print("\n--- Skipping Phase 1: Hyperparameter Optimization ---")

    # 2. Training and Cross-Validation Phase
    print("\n--- Phase 2: Model Training and Cross-Validation ---")
    train_main(config)

    # 3. Analysis Phase
    print("\n--- Phase 3: Analysis and Plotting ---")
    analysis = Analysis(config, config_path)
    analysis.run()

    print("\n--- Experiment Workflow Completed ---")


if __name__ == "__main__":
    main()
