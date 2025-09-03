"""
**REFACTORED**: This module defines the ReachabilityGraphDataModule,
which now uses the persistent InMemoryDataset classes for efficient data loading.
"""

from typing import Optional

import lightning.pytorch as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNAnalysisResultLabel
from src.SPNDatasets import ReachabilityGraphInMemoryDataset


class ReachabilityGraphDataModule(pl.LightningDataModule):
    """
    A LightningDataModule that uses pre-processed, on-disk datasets for
    SPN Reachability Graphs.
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
        super().__init__()
        self.save_hyperparameters()
        self.label_scaler = None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None):
        """Loads datasets and performs train/val/test splits."""
        if stage == "fit" or stage is None:
            full_dataset = ReachabilityGraphInMemoryDataset(
                self.hparams.root, self.hparams.train_file, self.hparams.label_to_predict
            )
            train_size = int(len(full_dataset) * (1 - self.hparams.val_split))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

            labels = torch.cat([data.y for data in self.train_dataset]).numpy().reshape(-1, 1)
            self.label_scaler = StandardScaler().fit(labels)
            self._scale_dataset(self.train_dataset)
            self._scale_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            self.test_dataset = ReachabilityGraphInMemoryDataset(
                self.hparams.root, self.hparams.test_file, self.hparams.label_to_predict
            )
            if self.label_scaler:
                self._scale_dataset(self.test_dataset)

    def _scale_dataset(self, dataset: Subset):
        """Applies the fitted scaler to the labels of a dataset."""
        for i in range(len(dataset)):
            data = dataset[i]
            y = data.y.numpy().reshape(-1, 1)
            data.y = torch.from_numpy(self.label_scaler.transform(y)).float().flatten()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
