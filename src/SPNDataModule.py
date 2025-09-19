"""This module defines the `SPNDataModule`, a flexible PyTorch Lightning
DataModule for handling SPN graph data.

It is designed to work with pre-loaded lists of `Data` or `HeteroData` objects,
correctly handles validation set creation (either from a split or a pre-defined
list), and exposes important dataset properties like the number of features.
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
    """A LightningDataModule that processes and serves SPN graph data.

    This module is designed to be flexible, accepting pre-loaded lists of graph
    data objects. It manages data splitting, label scaling, and provides
    dataloaders for training, validation, and testing. It also dynamically
    infers metadata about the dataset, such as feature dimensions.
    """

    def __init__(
        self,
        label_to_predict: SPNAnalysisResultLabel,
        train_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        val_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        test_data_list: Optional[List[Union[Data, HeteroData]]] = None,
        heterogeneous: bool = False,
        batch_size: int = 32,
        label_scaler: Optional[StandardScaler] = None,
        num_workers: int = 0,
        val_split: float = 0.2,
    ):
        """Initializes the SPNDataModule.

        Args:
            label_to_predict: The specific analysis result to use as the label.
            train_data_list: A list of data objects for training.
            val_data_list: An optional list of data objects for validation.
            test_data_list: An optional list of data objects for testing.
            heterogeneous: Whether the data is heterogeneous.
            batch_size: The batch size for the dataloaders.
            label_scaler: An optional pre-fitted StandardScaler for labels.
            num_workers: The number of workers for the dataloaders.
            val_split: The fraction of training data to use for validation if
                `val_data_list` is not provided.
        """
        super().__init__()
        self.save_hyperparameters("label_to_predict", "heterogeneous", "batch_size", "num_workers", "val_split")
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.test_data_list = test_data_list

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Dataset] = None
        self.label_scaler: Optional[StandardScaler] = label_scaler
        self._num_node_features: Union[int, Dict[str, int]] = 0
        self._num_edge_features: Union[int, Dict[str, int]] = 0
        self._num_node_types: int = 0
        self._num_edge_types: int = 0

    def setup(self, stage: Optional[str] = None):
        """Processes datasets, performs splits, and fits the label scaler.

        This method sets up the training, validation, and test datasets. If a
        label scaler is not provided, it fits one on the training data. It also
        infers and stores metadata about the dataset's structure.

        Args:
            stage: The stage for which to set up the data ('fit' or 'test').
        """
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

                # Fit scaler ONLY on the training split if it's not already provided
                if self.label_scaler is None:
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
        """Applies the fitted scaler to the labels of a dataset.

        Args:
            dataset: The dataset (or subset) whose labels need to be scaled.
        """
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
        """The number of node features in the dataset."""
        return self._num_node_features

    @property
    def num_edge_features(self) -> Union[int, Dict[str, int]]:
        """The number of edge features in the dataset."""
        return self._num_edge_features

    @property
    def num_node_types(self) -> int:
        """The number of node types in the dataset."""
        return self._num_node_types

    @property
    def num_edge_types(self) -> int:
        """The number of edge types in the dataset."""
        return self._num_edge_types

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
            drop_last=False,
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
