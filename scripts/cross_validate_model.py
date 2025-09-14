import argparse
import importlib
import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any

import lightning.pytorch as pl
import pandas as pd
import torch
from tqdm import tqdm

from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset
from src.config_utils import load_config


def load_model_dynamically(checkpoint_path: str) -> tuple[pl.LightningModule, dict]:
    """
    Dynamically loads a Lightning model from a .ckpt file.

    This is robust as it uses metadata saved by PyTorch Lightning to find the
    correct model class, avoiding "unexpected keys" errors.
    """
    # Ensure the project root is in the Python path to allow for dynamic imports
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    hparams = checkpoint.get("hyper_parameters", {})
    model_class_path = hparams.get("__pl_module_type_path__")

    if not model_class_path:
        raise KeyError(f"Could not find '__pl_module_type_path__' in checkpoint: {checkpoint_path}")

    try:
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import model class '{class_name}' from '{module_path}'. Error: {e}")

    # Load the model from the checkpoint file using the dynamically found class.
    return ModelClass.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu")), checkpoint


def cross_validate_models(experiment_dir: Path, output_file: Path, config: argparse.Namespace):
    """
    Finds all homogeneous model artifacts in an experiment directory, evaluates each against
    a specified set of pre-processed datasets, and aggregates results.
    """
    print(f"--- Scanning for model artifacts in: '{experiment_dir}' ---")
    checkpoint_files = sorted(list(experiment_dir.glob("**/best_model.ckpt")))

    if not checkpoint_files:
        print(f"Error: No 'best_model.ckpt' files found in any subdirectory of '{experiment_dir}'.")
        sys.exit(1)

    processed_data_dir = Path(config.io.root) / "processed"
    if not processed_data_dir.exists():
        print(f"Error: Processed data directory not found at '{processed_data_dir}'.")
        print("Please run the training script first to generate the processed data.")
        sys.exit(1)

    all_pt_files = {file.name: file for file in processed_data_dir.glob("*.pt")}
    if not all_pt_files:
        print(f"Error: No '.pt' files found in processed data directory '{processed_data_dir}'.")
        return

    print(f"Found {len(checkpoint_files)} trained model runs to evaluate.")

    all_results: List[Dict[str, Any]] = []

    for ckpt_path in tqdm(checkpoint_files, desc="Evaluating Model Runs"):
        try:
            model, checkpoint = load_model_dynamically(str(ckpt_path))
            model_hparams = checkpoint.get("hyper_parameters", {})
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
            label = datamodule_hparams.get("label_to_predict", config.model.label)

            # Determine the set of datasets to evaluate on
            target_datasets = config.cross_validation.datasets
            if target_datasets:
                print(f"--- Evaluating on {len(target_datasets)} specified datasets from config ---")
            else:
                print("--- Evaluating on all available datasets (excluding model's train/test sets) ---")

            # Exclude the model's own training and testing data
            excluded_files = {
                Path(model_hparams.get("train_file")).name,
                Path(model_hparams.get("test_file")).name,
            }

            for raw_fname in tqdm(target_datasets if target_datasets else sorted(all_pt_files.keys()), desc=f"Testing {ckpt_path.parent.name}", leave=False):
                sanitized_name = Path(raw_fname).stem
                if f"{sanitized_name}.processed" in excluded_files:
                    tqdm.write(f"Skipping {raw_fname} (used in training/testing of this model).")
                    continue

                pt_filename = f"data_{sanitized_name}_{label}.pt"
                if pt_filename not in all_pt_files:
                    warnings.warn(f"Could not find processed file '{pt_filename}' for raw file '{raw_fname}'. Skipping.")
                    continue

                test_dataset = HomogeneousSPNDataset(
                    root=str(config.io.root),
                    raw_data_dir=str(config.io.raw_data_dir),
                    raw_file_name=raw_fname,
                    label_to_predict=label,
                )

                if not test_dataset:
                    continue

                data_module = SPNDataModule(
                    test_data_list=list(test_dataset),
                    label_to_predict=label,
                    batch_size=datamodule_hparams.get("batch_size", 512),
                    num_workers=datamodule_hparams.get("num_workers", 0),
                )
                data_module.setup("test")

                trainer = pl.Trainer(
                    accelerator="auto",
                    devices="auto",
                    logger=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )
                test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

                if test_metrics:
                    result_dict = test_metrics[0]
                    result_dict["cross_eval_dataset"] = pt_filename
                    run_id_match = re.search(r"run_(\d+)", ckpt_path.parent.name)
                    seed_match = re.search(r"seed_(\d+)", ckpt_path.parent.name)
                    result_dict["gnn_operator"] = model_hparams.get("gnn_operator_name")
                    result_dict["run_id"] = int(run_id_match.group(1)) if run_id_match else -1
                    result_dict["seed"] = int(seed_match.group(1)) if seed_match else -1
                    all_results.append(result_dict)

        except Exception as e:
            warnings.warn(f"Failed to process run {ckpt_path.parent.name}. Reason: {e}")
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
    config, _ = load_config()

    experiment_dir = config.io.state_dict_dir / config.io.exp_name
    output_file = config.io.cross_eval_results_file

    cross_validate_models(experiment_dir=experiment_dir, output_file=output_file, config=config)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
