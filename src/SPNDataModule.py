"""
**REFACTORED**: This module defines the SPNDataModule, which now accepts
pre-loaded data lists, correctly handles validation set creation,
and exposes dataset properties like the number of features.
"""

from typing import Optional, Union, Dict, List

import lightning.pytorch as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split, Subset
from torch_geometric.data import Dataset, Data, HeteroData
from torch_geometric.loader import DataLoader

from src.PetriNets import SPNAnalysisResultLabel


class SPNDataModule(pl.LightningDataModule):
    """
    A LightningDataModule that processes lists of SPN graph data.
    It handles splitting, scaling, and batching.
    """

    def __init__(
        self,
        label_to_predict: SPNAnalysisResultLabel,
        train_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        val_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        test_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        heterogeneous: bool = False,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters("label_to_predict", "heterogeneous", "batch_size", "num_workers", "val_split")
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.test_data_list = test_data_list

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Dataset] = None
        self.label_scaler: Optional[StandardScaler] = None
        self._num_node_features: Union[int, Dict[str, int]] = 0
        self._num_edge_features: Union[int, Dict[str, int]] = 0
        self._num_node_types: int = 0
        self._num_edge_types: int = 0

    def setup(self, stage: Optional[str] = None):
        """Processes datasets and performs train/val/test splits."""
        if stage == "fit" or stage is None:
            if self.train_data_list:
                full_train_dataset = self.train_data_list

                if self.val_data_list is None:
                    train_size = int(len(full_train_dataset) * (1 - self.hparams.val_split))
                    val_size = len(full_train_dataset) - train_size
                    self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])
                else:
                    self.train_dataset = self.train_data_list
                    self.val_dataset = self.val_data_list

                # Fit scaler ONLY on the training split
                print("Fitting label scaler...")
                if self.hparams.heterogeneous:
                    labels = torch.cat([data["place"].y for data in self.train_dataset]).numpy().reshape(-1, 1)
                else:
                    labels = torch.cat([data.y for data in self.train_dataset]).numpy().reshape(-1, 1)
                self.label_scaler = StandardScaler().fit(labels)

                self._scale_dataset(self.train_dataset)
                self._scale_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            if self.test_data_list:
                self.test_dataset = self.test_data_list
                if self.label_scaler:
                    self._scale_dataset(self.test_dataset)

        # Capture metadata from the first available data object
        first_data = None
        if self.train_data_list:
            first_data = self.train_data_list[0]
        elif self.test_data_list:
            first_data = self.test_data_list[0]

        if first_data:
            if isinstance(first_data, HeteroData):
                self._num_node_features = {key: store.num_node_features for key, store in first_data.node_items()}
                self._num_edge_features = {key: store.num_edge_features for key, store in first_data.edge_items()}
                self._num_node_types = len(first_data.node_types)
                self._num_edge_types = len(first_data.edge_types)
            else:  # Homogeneous Data
                self._num_node_features = first_data.num_node_features
                self._num_edge_features = first_data.num_edge_features
                self._num_node_types = 1
                self._num_edge_types = 1

    def _scale_dataset(self, dataset: Union[Subset, List]):
        """Applies the fitted scaler to the labels of a dataset."""
        for i in range(len(dataset)):
            data = dataset[i]
            if self.hparams.heterogeneous:
                y = data["place"].y.numpy().reshape(-1, 1)
                data["place"].y = torch.from_numpy(self.label_scaler.transform(y)).float().flatten()
            else:
                y = data.y.numpy().reshape(-1, 1)
                data.y = torch.from_numpy(self.label_scaler.transform(y)).float().flatten()

    @property
    def num_node_features(self) -> Union[int, Dict[str, int]]:
        return self._num_node_features

    @property
    def num_edge_features(self) -> Union[int, Dict[str, int]]:
        return self._num_edge_features

    @property
    def num_node_types(self) -> int:
        return self._num_node_types

    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

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
