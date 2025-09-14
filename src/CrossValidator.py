import warnings
from pathlib import Path
from typing import List, Dict, Any

import lightning.pytorch as pl
import pandas as pd
from tqdm import tqdm

from src.SPNDataModule import SPNDataModule


class CrossValidator:
    """
    Evaluates a trained model against a collection of new or different datasets.
    """

    def __init__(self, data_loaders: List[SPNDataModule]):
        self.data_loaders = data_loaders

    def run(self, model: pl.LightningModule) -> pd.DataFrame:
        """
        Runs the cross-validation pipeline and returns a dataframe with the results.
        """
        all_results: List[Dict[str, Any]] = []

        for data_module in tqdm(self.data_loaders, desc="Cross-validating model", leave=False):
            try:
                model.eval()
                trainer = pl.Trainer(
                    accelerator="auto", logger=False, enable_progress_bar=False, enable_model_summary=False
                )
                test_metrics = trainer.test(model, datamodule=data_module, verbose=False)

                if test_metrics:
                    result_dict = test_metrics[0]
                    # The test dataset is a list of one graph, get its name from there
                    result_dict["cross_eval_dataset"] = data_module.test_dataset.data_list[0].raw_file_name
                    all_results.append(result_dict)

            except Exception as e:
                warnings.warn(f"Failed to process dataset for cross-validation. Reason: {e}")
                continue

        if all_results:
            return pd.DataFrame(all_results)

        return pd.DataFrame()
