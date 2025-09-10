import argparse
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any

import lightning.pytorch as pl
import pandas as pd
import torch
from tqdm import tqdm

from src.HomogeneousModels import (
    BaseGNN_SPN_Model,
    NodeGNN_SPN_Model,
    GraphGNN_SPN_Model,
    MixedGNN_SPN_Model,
)
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config


def setup_model(hparams: Dict[str, Any], node_features_dim: int) -> BaseGNN_SPN_Model:
    """
    Instantiates a GNN model from a dictionary of hyperparameters.
    """
    model_classes = {"node": NodeGNN_SPN_Model, "graph": GraphGNN_SPN_Model, "mixed": MixedGNN_SPN_Model}

    # This logic mirrors train_model.py for consistency
    model_key = hparams.get("gnn_operator")
    if model_key != "mixed":
        model_key = hparams.get("prediction_level")

    model_class = model_classes.get(model_key)
    if not model_class:
        raise ValueError(
            f"Could not find a model class for operator '{hparams.get('gnn_operator')}'"
            f" and prediction level '{hparams.get('prediction_level')}'."
        )

    model_kwargs = hparams.copy()
    if "num_layers_gnn" in model_kwargs:
        model_kwargs["num_layers"] = model_kwargs.pop("num_layers_gnn")

    # Filter kwargs to only those accepted by the model's constructor
    accepted_args = model_class.__init__.__code__.co_varnames
    filtered_kwargs = {k: v for k, v in model_kwargs.items() if k in accepted_args}

    # Ensure out_channels is set, as it's required by the model but not in hparams
    if "out_channels" not in filtered_kwargs:
        filtered_kwargs["out_channels"] = 1

    return model_class(node_features_dim=node_features_dim, **filtered_kwargs)


def cross_validate_models(experiment_dir: Path, data_dir: Path, output_file: Path, config: argparse.Namespace):
    """
    Finds all model artifacts in an experiment directory, evaluates each against
    all files in a data directory, and aggregates results.
    """
    print(f"--- Scanning for model artifacts in: '{experiment_dir}' ---")
    hparam_files = sorted(list(experiment_dir.glob("**/hparams.json")))
    if not hparam_files:
        print(f"Error: No 'hparams.json' files found in any subdirectory of '{experiment_dir}'.")
        return

    print(f"Found {len(hparam_files)} trained model runs to evaluate.")

    test_files = sorted(list(data_dir.glob("*.processed")))
    if not test_files:
        print(f"Error: No '.processed' files found in data directory '{data_dir}'.")
        return

    print(f"Found {len(test_files)} datasets to evaluate against.")

    all_results: List[Dict[str, Any]] = []

    for hparam_path in tqdm(hparam_files, desc="Evaluating Model Runs"):
        try:
            with open(hparam_path, "r") as f:
                hparams = json.load(f)

            model_path = hparam_path.parent / "best_model.pt"
            if not model_path.exists():
                warnings.warn(f"Skipping {hparam_path} because best_model.pt is missing.")
                continue

            for data_file in tqdm(test_files, desc=f"Testing {hparam_path.parent.name}", leave=False):
                # 3. Load a specific test dataset
                test_dataset = HomogeneousSPNDataset(
                    root=hparams.get("root", str(config.io.root)),
                    raw_data_dir=hparams.get("raw_data_dir", str(config.io.raw_data_dir)),
                    raw_file_name=data_file.name,
                    label_to_predict=hparams.get("label", config.model.label),
                )

                if not test_dataset:
                    continue

                # 4. Setup DataModule for evaluation
                data_module = SPNDataModule(
                    test_data_list=list(test_dataset),
                    label_to_predict=hparams.get("label", config.model.label),
                    batch_size=hparams.get("batch_size", 512),
                    num_workers=hparams.get("num_workers", 0),
                )
                # The scaler from the original training run is not available here.
                # The DataModule will create and fit a new one on the test data, which is acceptable for evaluation.
                data_module.setup("test")

                # 1. Reconstruct model from hyperparameters, now with node_features_dim
                model = setup_model(hparams, node_features_dim=data_module.num_node_features)

                # 2. Load the trained weights from the state dictionary
                model.load_state_dict(torch.load(model_path))
                model.eval()

                # 5. Run evaluation
                trainer = pl.Trainer(
                    accelerator="auto", logger=False, enable_progress_bar=False, enable_model_summary=False
                )
                test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

                # 6. Store results with metadata
                if test_metrics:
                    result_dict = test_metrics[0]
                    result_dict["cross_eval_dataset"] = data_file.name
                    # Add key hyperparameters for easy grouping during analysis
                    result_dict["gnn_operator"] = hparams.get("gnn_operator")
                    result_dict["run_id"] = hparams.get("run_id")
                    result_dict["seed"] = hparams.get("seed")
                    all_results.append(result_dict)

        except Exception as e:
            warnings.warn(f"Failed to process run {hparam_path.parent.name}. Reason: {e}")
            continue

    if all_results:
        print(f"\n--- Aggregating {len(all_results)} results and saving to {output_file} ---")
        results_df = pd.DataFrame(all_results)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_file, index=False)
        print("Cross-evaluation results saved successfully.")
    else:
        print("\n--- No successful evaluations were completed. No output file generated. ---")


def main():
    """Parses command-line arguments for the cross-validation script."""
    config = load_config()

    # All paths are now derived from the central config file
    experiment_dir = config.io.state_dict_dir / config.io.exp_name
    output_file = config.io.cross_eval_results_file
    data_dir = config.io.raw_data_dir

    cross_validate_models(
        experiment_dir=experiment_dir, data_dir=data_dir, output_file=output_file, config=config
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
