# test_model.py

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch

from src.PetriNets import load_spn_data_from_files

# We only need to import the DataModule, as it's used to process the new data.
# The model class will be imported dynamically.
from src.SPNDataModule import SPNDataModule


def load_model_dynamically(checkpoint_path: str) -> tuple[pl.LightningModule, dict]:
    """
    Dynamically loads a Lightning model and its checkpoint data.

    This function reads the model's class path from the checkpoint file,
    dynamically imports the necessary module, and then loads the model.

    Args:
        checkpoint_path (str): The path to the .ckpt file.

    Returns:
        A tuple containing:
        - The loaded and initialized LightningModule instance.
        - The raw checkpoint dictionary, useful for inspecting hyperparameters.
    """
    # Ensure the project's 'src' directory is in the Python path. This helps
    # the script find your custom modules when run from different locations.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # SECURITY NOTE: We are using `weights_only=False` which is the default but
    # generates a warning in recent PyTorch versions. This is necessary because
    # we need to load the full pickled object to access the 'hyper_parameters'
    # dictionary, which contains the model's class path for dynamic loading.
    #
    # **Only load checkpoints from a trusted source.**
    # Loading a checkpoint from an untrusted source with `weights_only=False`
    # can execute arbitrary malicious code.
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    model_class_path = hparams.get("__pl_module_type_path__")

    if not model_class_path:
        raise KeyError(
            "Could not find model class path in checkpoint. "
            "Ensure the model was saved with `save_hyperparameters()`."
        )

    try:
        module_path, class_name = model_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ModelClass = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import model class '{class_name}' from '{module_path}'. "
            f"Ensure the module is in your PYTHONPATH. Original error: {e}"
        )

    # Load the model using the dynamically found class.
    # `load_from_checkpoint` will handle loading the weights into the model instance.
    model = ModelClass.load_from_checkpoint(checkpoint_path, map_location=torch.device("cpu"))
    return model, checkpoint


def evaluate_model_on_directory(checkpoint_path: str, data_directory: str, output_tsv_path: str):
    """
    Loads a trained model and evaluates it against all valid .processed files
    in a given directory, writing the results to a TSV file.
    """
    print(f"--- Loading model dynamically from checkpoint: {checkpoint_path} ---")

    # --- 1. Load Model and Hyperparameters Dynamically ---
    try:
        model, checkpoint = load_model_dynamically(checkpoint_path)
        model.eval()  # Set model to evaluation mode
        datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})
    except (FileNotFoundError, KeyError, ImportError) as e:
        print(f"Error: Failed to load the model. Reason: {e}")
        return

    # Extract necessary hparams for the datamodule
    label_to_predict = datamodule_hparams.get("label_to_predict")
    if not label_to_predict:
        print("Error: 'label_to_predict' not found in checkpoint. Cannot proceed.")
        return
    batch_size = datamodule_hparams.get("batch_size", 128)
    num_workers = datamodule_hparams.get("num_workers", 0)

    # --- 2. Find and Iterate Through Test Files ---
    all_results: List[Dict[str, Any]] = []
    corrupted_files: List[Dict[str, str]] = []
    test_files = sorted(list(Path(data_directory).glob("*.processed")))

    if not test_files:
        print(f"Error: No '.processed' files found in directory '{data_directory}'.")
        return

    print(f"\nFound {len(test_files)} files to test. Starting evaluation...")

    for data_file in test_files:
        print(f"  -> Testing on: {data_file.name}")
        try:
            # --- 3. Load and Prepare Data for a Single File ---
            new_test_data = load_spn_data_from_files(data_file)
            if not new_test_data:
                raise ValueError("Loaded data is empty or invalid.")

            data_module = SPNDataModule(
                label_to_predict=label_to_predict,
                train_data_list=[],
                val_data_list=[],
                test_data_list=new_test_data,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # --- 4. Run Test ---
            trainer = pl.Trainer(
                accelerator="auto",
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

            if test_metrics:
                result_dict = test_metrics[0]
                result_dict["filename"] = data_file.name
                all_results.append(result_dict)
            else:
                raise RuntimeError("Trainer.test() produced no metrics.")

        except Exception as e:
            corrupted_files.append({"filename": data_file.name, "error": str(e)})
            print(f"     [SKIPPED] Could not process file. Reason: {e}")

    # --- 5. Write Aggregated Results to TSV File ---
    if all_results:
        first_result_keys = list(all_results[0].keys())
        first_result_keys.remove("filename")
        fieldnames = ["filename"] + sorted(first_result_keys)

        print(f"\n--- Writing {len(all_results)} results to {output_tsv_path} ---")
        with open(output_tsv_path, "w", newline="", encoding="utf-8") as tsvfile:
            writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(all_results)
    else:
        print("\n--- No successful tests were completed. No output file generated. ---")

    # --- 6. Report Corrupted Files ---
    if corrupted_files:
        print("\n--- The following files failed to process: ---")
        for item in corrupted_files:
            print(f"- {item['filename']}: {item['error']}")


def get_test_args():
    """Parses command-line arguments for the test script."""
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN model on a directory of SPN data.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the model checkpoint (.ckpt) file.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./Data",
        help="Directory containing the .processed test files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_results.tsv",
        help="Path to the output TSV file for the results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = get_test_args()

    evaluate_model_on_directory(
        checkpoint_path=args.checkpoint_path,
        data_directory=args.data_dir,
        output_tsv_path=args.output_file,
    )
