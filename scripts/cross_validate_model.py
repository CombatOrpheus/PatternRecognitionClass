import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import pandas as pd
import torch
from tqdm import tqdm

from src.config_utils import load_config
from src.path_utils import PathHandler
from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset


def load_model_dynamically(checkpoint_path: str) -> tuple[pl.LightningModule, dict]:
    """
    Dynamically loads a Lightning model from a .ckpt file.
    """
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

    return ModelClass.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu")), checkpoint


def cross_validate_models(paths: PathHandler, config: dict):
    """
    Finds all model artifacts, evaluates each against all data files, and aggregates results.
    """
    experiment_dir = paths.io_config.state_dict_dir / config.io.exp_name
    print(f"--- Scanning for model artifacts in: '{experiment_dir}' ---")
    checkpoint_files = paths.find_model_checkpoints(experiment_dir)

    if not checkpoint_files:
        print(f"Error: No 'best_model.ckpt' files found in any subdirectory of '{experiment_dir}'.")
        sys.exit(1)

    test_files = paths.find_processed_data_files(paths.io_config.raw_data_dir)
    if not test_files:
        print(f"Error: No '.processed' files found in data directory '{paths.io_config.raw_data_dir}'.")
        return

    print(f"Found {len(checkpoint_files)} trained model runs to evaluate.")

    all_results: List[Dict[str, Any]] = []

    for ckpt_path in tqdm(checkpoint_files, desc="Evaluating Model Runs"):
        try:
            model, checkpoint = load_model_dynamically(str(ckpt_path))
            model_hparams = checkpoint.get("hyper_parameters", {})
            datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
            path_metadata = PathHandler.parse_metadata_from_path(ckpt_path)

            for data_file in tqdm(test_files, desc=f"Testing {ckpt_path.parent.name}", leave=False):
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
                    result_dict["cross_eval_dataset"] = data_file.name
                    result_dict.update(path_metadata)
                    all_results.append(result_dict)

        except Exception as e:
            warnings.warn(f"Failed to process run {ckpt_path.parent.name}. Reason: {e}")
            continue

    if all_results:
        output_file = paths.get_cross_eval_results_path()
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
    paths = PathHandler(config.io)
    cross_validate_models(paths=paths, config=config)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
