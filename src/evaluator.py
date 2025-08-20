from pathlib import Path
from typing import List, Dict, Any

import lightning.pytorch as pl
from tqdm import tqdm

from src.PetriNets import load_spn_data_from_files
from src.SPNDataModule import SPNDataModule


class ExperimentEvaluator:
    """
    A class to evaluate a trained model against a suite of test datasets.
    It pre-loads and caches all test datasets upon initialization for efficiency.
    """

    def __init__(self, data_directory: Path):
        """
        Initializes the evaluator, finds all test files, and pre-loads them
        into an in-memory cache.
        """
        self.data_directory = data_directory
        self.test_files = self._find_test_files()
        # --- NEW: Pre-load and cache all test data ---
        self.cached_data = self._cache_test_data()

    def _find_test_files(self) -> List[Path]:
        """Finds all '.processed' data files in the data directory."""
        print(f"\n--- Finding test datasets in: {self.data_directory} ---")
        test_files = sorted(list(self.data_directory.glob("*.processed")))
        if not test_files:
            print(f"Warning: No '.processed' files found in cross-evaluation data directory '{self.data_directory}'.")
        else:
            print(f"Found {len(test_files)} test datasets for evaluation.")
        return test_files

    def _cache_test_data(self) -> Dict[Path, List[Any]]:
        """Loads all found test files into an in-memory dictionary."""
        if not self.test_files:
            return {}

        print("--- Pre-loading and caching all test datasets for evaluation ---")
        cached_data = {}
        for data_file in tqdm(self.test_files, desc="Caching test data"):
            try:
                # Note: SPNDataModule will handle the conversion to PyG Data objects later
                raw_data = load_spn_data_from_files(data_file)
                if raw_data:
                    cached_data[data_file] = raw_data
            except Exception as e:
                print(f"Warning: Could not load or cache file {data_file.name}. Skipping. Error: {e}")
        return cached_data

    def evaluate(self, model: pl.LightningModule, hparams: Dict, datamodule_hparams: Dict) -> List[Dict]:
        """
        Evaluates a given model against all pre-loaded test datasets.
        """
        if not self.cached_data:
            return []

        model.eval()
        all_eval_results = []

        label_to_predict = datamodule_hparams.get("label_to_predict")
        batch_size = datamodule_hparams.get("batch_size", 128)
        num_workers = datamodule_hparams.get("num_workers", 0)

        for data_file, new_test_data in tqdm(self.cached_data.items(), desc="Cross-Dataset Evaluation", leave=False):
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
                result_dict["cross_eval_dataset"] = data_file.name
                for param, value in hparams.items():
                    if isinstance(value, (str, int, float, bool)):
                        result_dict[param] = value
                all_eval_results.append(result_dict)

        return all_eval_results
