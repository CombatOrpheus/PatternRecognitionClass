"""This module defines the `ReachabilityGraphDataModule`, a PyTorch Lightning
DataModule for handling SPN Reachability Graph datasets.

It leverages the `ReachabilityGraphInMemoryDataset` for efficient data loading
and preprocessing, including train/validation/test splitting and label scaling.
"""

from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNAnalysisResultLabel
from src.SPNDatasets import ReachabilityGraphInMemoryDataset


class ReachabilityGraphDataModule(pl.LightningDataModule):
    """A LightningDataModule for SPN Reachability Graph datasets.

    This class handles the loading, splitting, and preprocessing of reachability
    graph data. It uses an in-memory dataset for efficiency and applies label
    scaling to normalize the target values.
    """

    def __init__(
        self,
        root: str,
        train_file: str,
        val_file: str,
        test_file: str,
        label_to_predict: SPNAnalysisResultLabel,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        """Initializes the ReachabilityGraphDataModule.

        Args:
            root: The root directory where the dataset should be stored.
            train_file: The name of the training data file.
            val_file: The name of the validation data file.
            test_file: The name of the test data file.
            label_to_predict: The specific analysis result to use as the label.
            batch_size: The batch size for the dataloaders.
            num_workers: The number of workers for the dataloaders.
            val_split: The fraction of the training data to use for validation.
        """
        super().__init__()
        self.save_hyperparameters()
        self.label_scaler = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None):
        """Loads datasets, performs splits, and fits the label scaler.

        This method is called by PyTorch Lightning. It sets up the training,
        validation, and test datasets based on the provided stage.

        Args:
            stage: The stage for which to set up the data ('fit' or 'test').
        """
        if stage == "fit" or stage is None:
            train_path = Path(self.hparams.train_file)
            full_dataset = ReachabilityGraphInMemoryDataset(
                root=self.hparams.root,
                raw_data_dir=train_path.parent,
                raw_file_name=train_path.name,
                label_to_predict=self.hparams.label_to_predict,
            )
            train_size = int(len(full_dataset) * (1 - self.hparams.val_split))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

            labels = torch.cat([data.y for data in self.train_dataset]).numpy().reshape(-1, 1)
            self.label_scaler = StandardScaler().fit(labels)
            self._scale_dataset(self.train_dataset)
            self._scale_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            test_path = Path(self.hparams.test_file)
            self.test_dataset = ReachabilityGraphInMemoryDataset(
                root=self.hparams.root,
                raw_data_dir=test_path.parent,
                raw_file_name=test_path.name,
                label_to_predict=self.hparams.label_to_predict,
            )
            if self.label_scaler:
                self._scale_dataset(self.test_dataset)

    def _scale_dataset(self, dataset: Subset):
        """Applies the fitted scaler to the labels of a dataset subset.

        Args:
            dataset: The dataset subset whose labels need to be scaled.
        """
        for i in range(len(dataset)):
            data = dataset[i]
            y = data.y.numpy().reshape(-1, 1)
            data.y = torch.from_numpy(self.label_scaler.transform(y)).float().flatten()

    def train_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the training set.

        Returns:
            The training DataLoader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the validation set.

        Returns:
            The validation DataLoader.
        """
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the test set.

        Returns:
            The test DataLoader.
        """
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
