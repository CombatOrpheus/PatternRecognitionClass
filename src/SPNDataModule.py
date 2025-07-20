"""
This module defines the SPNDataModule, a PyTorch Lightning DataModule for
handling and preparing Stochastic Petri Net (SPN) data for GNN models.

This version is designed to accept pre-split training, validation, and test sets.
"""
from typing import Optional, List

import pytorch_lightning as pl
from torch import from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNData, SPNAnalysisResultLabel


class SPNDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for creating training, validation, and test datasets
    for Stochastic Petri Nets from pre-split data sources.

    This class handles:
    - Accepting raw, pre-split SPN data lists.
    - Converting them into a PyTorch Geometric `Data` format.
    - Providing `DataLoader` instances for each split.
    """

    def __init__(
            self,
            label_to_predict: SPNAnalysisResultLabel,
            train_data_list: List[SPNData],
            val_data_list: Optional[List[SPNData]] = None,
            test_data_list: Optional[List[SPNData]] = None,
            batch_size: int = 32,
            num_workers: int = 0,
    ):
        """
        Args:
            label_to_predict (SPNAnalysisResultLabel): The specific analysis result
                to use as the ground truth label (y) for the models.
            train_data_list (List[SPNData]): A list of raw SPNData objects for training.
            val_data_list (Optional[List[SPNData]]): An optional list for validation.
            test_data_list (Optional[List[SPNData]]): An optional list for testing.
            batch_size (int): The batch size for the DataLoaders.
            num_workers (int): The number of subprocesses to use for data loading.
        """
        super().__init__()
        # Store hyperparameters, ignoring the large data lists themselves
        self.save_hyperparameters(ignore=['train_data_list', 'val_data_list', 'test_data_list'])

        # Store the raw data lists
        self.train_raw = train_data_list
        self.val_raw = val_data_list
        self.test_raw = test_data_list

        # Datasets will be populated in setup()
        self.train_data: Optional[List[Data]] = None
        self.val_data: Optional[List[Data]] = None
        self.test_data: Optional[List[Data]] = None

        self.label_to_predict = label_to_predict
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _process_list(self, raw_data_list: List[SPNData]) -> List[Data]:
        """Converts a list of raw SPNData objects to PyG Data objects."""
        processed_data = []
        for net in raw_data_list:
            node_features, edge_features, edge_pairs = net.to_information()
            label = net.get_analysis_result(self.hparams.label_to_predict)

            processed_data.append(
                Data(
                    x=from_numpy(node_features).float(),
                    edge_attr=from_numpy(edge_features).float(),
                    edge_index=from_numpy(edge_pairs).long(),
                    y=from_numpy(label)
                )
            )
        return processed_data

    def setup(self, stage: Optional[str] = None):
        """
        This method is called by Lightning to prepare the data. It processes
        the raw data lists into PyG Data objects.
        """
        # The 'stage' hint allows us to only process data when needed.
        if (stage == "fit" or stage is None) and self.train_raw:
            self.train_data = self._process_list(self.train_raw)
            if self.val_raw:
                self.val_data = self._process_list(self.val_raw)

        if (stage == "test" or stage is None) and self.test_raw:
            self.test_data = self._process_list(self.test_raw)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the validation set."""
        if not self.val_data:
            return None  # Lightning will skip validation if this is None
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the test set."""
        if not self.test_data:
            return None  # Lightning will skip testing if this is None
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
