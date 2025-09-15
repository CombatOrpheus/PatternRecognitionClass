"""This script provides a robust mechanism for evaluating trained model
checkpoints against a directory of test datasets.

It is designed to be run after model training and is capable of:
-   Dynamically loading different model architectures from their checkpoints.
-   Inferring metadata (like experiment name and run ID) from file paths.
-   Iterating through all checkpoints in a specified experiment directory.
-   Evaluating each checkpoint against every `.processed` file in a data directory.
-   Aggregating all evaluation metrics into a single Parquet file for analysis.
"""
import argparse
import importlib
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import lightning.pytorch as pl
import pandas as pd
import torch
from tqdm import tqdm

from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule
from src.config_utils import load_config


def load_model_dynamically(checkpoint_path: str) -> Tuple[pl.LightningModule, dict]:
    """Dynamically loads a Lightning model and its checkpoint data.

    This function reads the model's class path from the checkpoint's
    hyperparameters, imports the module, and loads the model from the
    checkpoint.

    Args:
        checkpoint_path: The path to the model checkpoint file.

    Returns:
        A tuple containing the loaded LightningModule and the checkpoint data.

    Raises:
        KeyError: If the model class path is not found in the checkpoint.
        ImportError: If the model class cannot be imported.
    """
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    model_class_path = hparams.get("__pl_module_type_path__")

    if not model_class_path:
        raise KeyError("Could not find model class path in checkpoint.")

    try:
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import model class '{class_name}' from '{module_path}'. Error: {e}")

    model = ModelClass.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    return model, checkpoint


def parse_metadata_from_path(ckpt_path: Path, experiment_dir: Path) -> Dict[str, Any]:
    """Infers metadata from the checkpoint's file path.

    This function extracts the experiment name, GNN operator, run ID, and seed
    from the directory structure.

    Args:
        ckpt_path: The path to the model checkpoint file.
        experiment_dir: The root directory of the experiment.

    Returns:
        A dictionary containing the inferred metadata.
    """
    metadata = {}

    # 1. Get experiment name from the parent directory
    metadata["experiment_name"] = experiment_dir.name

    # 2. Infer GNN operator from the experiment name
    known_operators = ["gcn", "tag", "cheb", "sgc", "ssg"]
    name_parts = experiment_dir.name.split("_")
    found_operator = [op for op in known_operators if op in name_parts]
    metadata["gnn_operator_inferred"] = found_operator[0] if found_operator else "unknown"

    # 3. Parse run_id and seed from the version directory (e.g., 'run_0_seed_42')
    version_dir_name = ckpt_path.parent.parent.name
    match = re.match(r"run_(\d+)_seed_(\d+)", version_dir_name)
    if match:
        metadata["run_id_inferred"] = int(match.group(1))
        metadata["seed_inferred"] = int(match.group(2))

    return metadata


def evaluate_experiment_on_directory(experiment_dir: Path, data_directory: Path, output_parquet_path: Path):
    """Evaluates all model checkpoints in a directory against a set of datasets.

    It scans the `experiment_dir` for all `.ckpt` files, preferring `best.ckpt`
    if available for a given run. Each found checkpoint is then tested against
    every `.processed` file in the `data_directory`. The results are compiled
    into a pandas DataFrame and saved as a Parquet file.

    Args:
        experiment_dir: The directory containing the experiment's results and
            checkpoints.
        data_directory: The directory containing the `.processed` test datasets.
        output_parquet_path: The path to save the aggregated results Parquet file.
    """
    print(f"--- Scanning for model checkpoints in: {experiment_dir} ---")
    all_ckpts = sorted(list(experiment_dir.glob("**/checkpoints/*.ckpt")))

    # Group checkpoints by their parent directory to handle multiple files per run
    checkpoints_by_dir: Dict[Path, List[Path]] = {}
    for ckpt_path in all_ckpts:
        dir_path = ckpt_path.parent
        if dir_path not in checkpoints_by_dir:
            checkpoints_by_dir[dir_path] = []
        checkpoints_by_dir[dir_path].append(ckpt_path)

    # Select the definitive checkpoint for each run directory
    final_checkpoint_paths = []
    for dir_path, ckpts in checkpoints_by_dir.items():
        best_ckpt_path = dir_path / "best.ckpt"
        # Check for existence of 'best.ckpt' in the list of found checkpoints
        if best_ckpt_path in ckpts:
            final_checkpoint_paths.append(best_ckpt_path)  # Prefer 'best.ckpt'
        elif ckpts:
            # If no 'best.ckpt', take the first available checkpoint.
            # This is useful if only one checkpoint is saved without being named 'best'.
            final_checkpoint_paths.append(ckpts[0])

    checkpoint_paths = sorted(final_checkpoint_paths)

    if not checkpoint_paths:
        print(f"Error: No '.ckpt' files found in any 'checkpoints' subdirectory of '{experiment_dir}'.")
        return

    print(f"Found {len(checkpoint_paths)} model checkpoints to evaluate.")

    test_files = sorted(list(data_directory.glob("*.processed")))
    if not test_files:
        print(f"Error: No '.processed' files found in data directory '{data_directory}'.")
        return

    print(f"Found {len(test_files)} test datasets to evaluate against.")

    all_results: List[Dict[str, Any]] = []

    for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating Checkpoints"):
        try:
            # --- NEW: Parse metadata from the path structure ---
            path_metadata = parse_metadata_from_path(ckpt_path, experiment_dir)

            model, checkpoint = load_model_dynamically(str(ckpt_path))
            model.eval()
            model_hparams = checkpoint.get("hyper_parameters", {})
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})

            label_to_predict = datamodule_hparams.get("label_to_predict")
            batch_size = datamodule_hparams.get("batch_size", 128)
            num_workers = datamodule_hparams.get("num_workers", 0)

            for data_file in tqdm(test_files, desc=f"Testing {ckpt_path.parent.parent.name}", leave=False):
                new_test_data = load_spn_data_from_files(data_file)
                if not new_test_data:
                    continue

                data_module = SPNDataModule(
                    label_to_predict=label_to_predict,
                    train_data_list=[],
                    val_data_list=[],
                    test_data_list=new_test_data,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )

                trainer = pl.Trainer(
                    accelerator="auto", logger=False, enable_progress_bar=False, enable_model_summary=False
                )
                test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

                if test_metrics:
                    result_dict = test_metrics[0]
                    result_dict["test_dataset"] = data_file.name

                    # Add path-based and hyperparameter metadata
                    result_dict.update(path_metadata)
                    for param, value in model_hparams.items():
                        if isinstance(value, (str, int, float, bool)):
                            result_dict[param] = value
                    all_results.append(result_dict)

        except Exception as e:
            print(f"Warning: Failed to process checkpoint {ckpt_path}. Reason: {e}")
            continue

    if all_results:
        print(f"\n--- Aggregating {len(all_results)} results and saving to {output_parquet_path} ---")
        results_df = pd.DataFrame(all_results)

        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_parquet_path, index=False)
        print("Evaluation results saved successfully.")
    else:
        print("\n--- No successful evaluations were completed. No output file generated. ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config, _ = load_config()

    evaluate_experiment_on_directory(
        experiment_dir=config.io.experiment_dir,
        data_directory=config.io.data_dir,
        output_parquet_path=config.io.output_file,
    )
