import importlib
import sys
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


def load_model_dynamically(checkpoint_path: str) -> tuple[pl.LightningModule, dict]:
    """
    Dynamically loads a Lightning model and its checkpoint data.
    """
    project_root = Path(__file__).resolve().parent.parent
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


def evaluate_experiment_on_directory(paths: PathHandler):
    """
    Finds all model checkpoints, evaluates them against all test data, and aggregates results.
    """
    experiment_dir = paths.io_config.experiment_dir
    print(f"--- Scanning for model checkpoints in: {experiment_dir} ---")
    checkpoint_paths = paths.find_model_checkpoints(experiment_dir)

    if not checkpoint_paths:
        print(f"Error: No 'best.ckpt' files found in any 'checkpoints' subdirectory of '{experiment_dir}'.")
        return

    print(f"Found {len(checkpoint_paths)} model checkpoints to evaluate.")

    test_files = paths.find_processed_data_files()
    if not test_files:
        print(f"Error: No '.processed' files found in data directory '{paths.io_config.data_dir}'.")
        return

    print(f"Found {len(test_files)} test datasets to evaluate against.")

    all_results: List[Dict[str, Any]] = []

    for ckpt_path in tqdm(checkpoint_paths, desc="Evaluating Checkpoints"):
        try:
            path_metadata = PathHandler.parse_metadata_from_path(ckpt_path)

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
                    result_dict.update(path_metadata)
                    for param, value in model_hparams.items():
                        if isinstance(value, (str, int, float, bool)):
                            result_dict[param] = value
                    all_results.append(result_dict)

        except Exception as e:
            print(f"Warning: Failed to process checkpoint {ckpt_path}. Reason: {e}")
            continue

    if all_results:
        output_path = paths.get_cross_eval_results_path()
        print(f"\n--- Aggregating {len(all_results)} results and saving to {output_path} ---")
        results_df = pd.DataFrame(all_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        print("Evaluation results saved successfully.")
    else:
        print("\n--- No successful evaluations were completed. No output file generated. ---")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    config, _ = load_config()
    path_handler = PathHandler(config.io)
    evaluate_experiment_on_directory(path_handler)
