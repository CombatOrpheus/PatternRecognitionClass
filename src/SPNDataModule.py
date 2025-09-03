"""
**REFACTORED**: This module defines the SPNDataModule, which now uses the
persistent InMemoryDataset classes and correctly handles validation set creation.
"""

from typing import Optional

import lightning.pytorch as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNAnalysisResultLabel
from src.SPNDatasets import HomogeneousSPNDataset, HeterogeneousSPNDataset


class SPNDataModule(pl.LightningDataModule):
    """
    A LightningDataModule that uses pre-processed, on-disk datasets for SPNs.
    """

    def __init__(
        self,
        root: str,
        raw_data_dir: str,
        train_file: str,
        test_file: str,
        label_to_predict: SPNAnalysisResultLabel,
        val_file: Optional[str] = None,  # **BUG FIX**: Made optional
        heterogeneous: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None
        self.label_scaler: Optional[StandardScaler] = None

    def setup(self, stage: Optional[str] = None):
        """Loads datasets and performs train/val/test splits."""
        dataset_class = HeterogeneousSPNDataset if self.hparams.heterogeneous else HomogeneousSPNDataset

        if stage == "fit" or stage is None:
            # **BUG FIX**: Handle both provided val_file and automatic splitting
            if self.hparams.val_file is None:
                # Create validation set from a split of the training data
                full_train_dataset = dataset_class(
                    self.hparams.root, self.hparams.raw_data_dir, self.hparams.train_file, self.hparams.label_to_predict
                )
                train_size = int(len(full_train_dataset) * (1 - self.hparams.val_split))
                val_size = len(full_train_dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])
            else:
                # Use separate files for training and validation
                self.train_dataset = dataset_class(
                    self.hparams.root, self.hparams.raw_data_dir, self.hparams.train_file, self.hparams.label_to_predict
                )
                self.val_dataset = dataset_class(
                    self.hparams.root, self.hparams.raw_data_dir, self.hparams.val_file, self.hparams.label_to_predict
                )

            # Fit scaler ONLY on the training split
            print("Fitting label scaler...")
            if self.hparams.heterogeneous:
                labels = torch.cat([data["place"].y for data in self.train_dataset]).numpy().reshape(-1, 1)
            else:
                labels = torch.cat([data.y for data in self.train_dataset]).numpy().reshape(-1, 1)
            self.label_scaler = StandardScaler().fit(labels)

            # Apply scaling in-place
            self._scale_dataset(self.train_dataset)
            self._scale_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            self.test_dataset = dataset_class(
                self.hparams.root, self.hparams.raw_data_dir, self.hparams.test_file, self.hparams.label_to_predict
            )
            if self.label_scaler:
                self._scale_dataset(self.test_dataset)

    def _scale_dataset(self, dataset: Subset):
        """Applies the fitted scaler to the labels of a dataset."""
        for i in range(len(dataset)):
            data = dataset[i]
            if self.hparams.heterogeneous:
                y = data["place"].y.numpy().reshape(-1, 1)
                data["place"].y = torch.from_numpy(self.label_scaler.transform(y)).float().flatten()
            else:
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
