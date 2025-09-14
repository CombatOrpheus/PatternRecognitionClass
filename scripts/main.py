import torch

from src.config_utils import load_config
from src.Analysis import Analysis
from scripts.train_model import main as train_main
from scripts.optimize_hyperparameters import main as optimize_main


def main():
    """
    Main function to orchestrate the entire experiment workflow, including:
    1. Hyperparameter optimization.
    2. Training of models and immediate cross-validation.
    3. Analysis and plotting of the results.
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
