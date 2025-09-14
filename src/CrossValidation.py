import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any

import lightning.pytorch as pl
from tqdm import tqdm

from src.SPNDataModule import SPNDataModule
from src.SPNDatasets import HomogeneousSPNDataset


class CrossValidator:
    def __init__(self, config):
        self.config = config

    def _get_datasets_to_evaluate(self) -> List[Path]:
        """
        Determines the list of raw dataset files to use for cross-validation.
        """
        # If a specific list of datasets is provided in the config, use that.
        if self.config.cross_validation.datasets:
            return [Path(self.config.io.raw_data_dir) / f for f in self.config.cross_validation.datasets]

        # Otherwise, get all files from the raw data directory.
        raw_data_dir = Path(self.config.io.raw_data_dir)
        if not raw_data_dir.exists():
            warnings.warn(f"Raw data directory not found: {raw_data_dir}. Cannot perform cross-validation.")
            return []

        # Return all files, letting the caller handle filtering.
        return sorted(list(raw_data_dir.iterdir()))

    def cross_validate_single_model(
        self, model: pl.LightningModule, run_config: argparse.Namespace, run_id: int, seed: int
    ) -> List[Dict[str, Any]]:
        """
        Evaluates a single trained model instance against a specified set of datasets.
        """
        results: List[Dict[str, Any]] = []
        model_hparams = model.hparams
        label = model_hparams.get("label_to_predict", self.config.model.label)

        # Determine which datasets this model was trained/tested on so we can exclude them.
        excluded_raw_files = {
            Path(run_config.train_file),
            Path(run_config.test_file),
        }

        datasets_to_eval = self._get_datasets_to_evaluate()

        desc = f"Cross-validating run {run_id} ({run_config.gnn_operator_name})"
        for raw_path in tqdm(datasets_to_eval, desc=desc, leave=False):
            if raw_path.name in excluded_raw_files:
                continue

            try:
                # Let the Dataset class handle finding/creating the processed file.
                test_dataset = HomogeneousSPNDataset(
                    root=str(self.config.io.root),
                    raw_data_dir=str(self.config.io.raw_data_dir),
                    raw_file_name=raw_path.name,
                    label_to_predict=label,
                )
                if not test_dataset or len(test_dataset) == 0:
                    warnings.warn(f"Skipping empty or invalid dataset: {raw_path.name}")
                    continue

                data_module = SPNDataModule(
                    test_data_list=list(test_dataset),
                    label_to_predict=label,
                    batch_size=run_config.batch_size,
                    num_workers=self.config.training.num_workers,
                )
                data_module.setup("test")

                trainer = pl.Trainer(accelerator="auto", devices="auto", logger=False, enable_progress_bar=False)
                test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

                if test_metrics:
                    result_dict = test_metrics[0]
                    # The processed file name is now an attribute of the dataset
                    result_dict["cross_eval_dataset"] = test_dataset.processed_file_names
                    result_dict["gnn_operator"] = run_config.gnn_operator_name
                    result_dict["run_id"] = run_id
                    result_dict["seed"] = seed
                    results.append(result_dict)

            except Exception as e:
                warnings.warn(f"Failed to cross-validate on {raw_path.name} for run {run_id}. Reason: {e}")
                continue

        return results
