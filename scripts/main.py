import torch

from src.config_utils import load_config
from src.Analysis import Analysis
from scripts.train_model import main as train_main


def main():
    """
    Main function to orchestrate the entire experiment workflow, including:
    1. Training of models and immediate cross-validation.
    2. Analysis and plotting of the results.
    """
    torch.set_float32_matmul_precision("high")

    print("--- Starting Experiment Workflow ---")

    # 1. Training and Cross-Validation Phase
    print("\n--- Phase 1: Model Training and Cross-Validation ---")
    train_main()

    # 2. Analysis Phase
    print("\n--- Phase 2: Analysis and Plotting ---")
    config, _ = load_config()
    analysis = Analysis(config)
    analysis.run()

    print("\n--- Experiment Workflow Completed ---")


if __name__ == "__main__":
    main()
