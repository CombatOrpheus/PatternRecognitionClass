"""
This module defines the ReachabilityGraphDataModule, a PyTorch Lightning
DataModule for handling the reachability graphs of Stochastic Petri Nets (SPNs).

It correctly processes the reachability graph, where nodes represent markings
and edges represent transitions, making it suitable for dynamic analysis with GNNs.
"""

from typing import Optional, List, Callable

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.PetriNets import SPNData, SPNAnalysisResultLabel


class ReachabilityGraphDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for creating datasets from SPN reachability graphs.
    """

    def __init__(
        self,
        label_to_predict: SPNAnalysisResultLabel,
        train_data_list: List[SPNData],
        val_data_list: Optional[List[SPNData]] = None,
        test_data_list: Optional[List[SPNData]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_data_list", "val_data_list", "test_data_list"])

        self.train_raw = train_data_list
        self.val_raw = val_data_list
        self.test_raw = test_data_list

        self.train_data: Optional[List[Data]] = None
        self.val_data: Optional[List[Data]] = None
        self.test_data: Optional[List[Data]] = None

        self.transform = transform
        self.label_scaler: Optional[StandardScaler] = None
        self.max_num_places: Optional[int] = None  # To store max feature dimension

    def _process_list(self, raw_data_list: List[SPNData], desc: Optional[str] = None) -> List[Data]:
        """Converts a list of raw SPNData objects to PyG Data objects with padding."""
        processed_data = []
        for net in tqdm(raw_data_list, leave=False, desc=desc):
            node_features = torch.tensor(np.array(net.reachability_graph_nodes), dtype=torch.float)

            if self.max_num_places is not None:
                padding_size = self.max_num_places - node_features.shape[1]
                if padding_size > 0:
                    node_features = F.pad(node_features, (0, padding_size), "constant", 0)

            edge_index = torch.tensor(net.reachability_graph_edges, dtype=torch.long).t().contiguous()
            firing_rates = torch.tensor(net.average_firing_rates, dtype=torch.float)
            transition_indices = torch.tensor(net.transition_indices, dtype=torch.long)
            edge_attr = firing_rates[transition_indices].view(-1, 1)

            label_raw = net.get_analysis_result(self.hparams.label_to_predict)
            if not isinstance(label_raw, np.ndarray):
                label_raw = np.array([label_raw])

            label_scaled = self.label_scaler.transform(label_raw.reshape(-1, 1)) if self.label_scaler else label_raw

            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.from_numpy(label_scaled).float().flatten(),
            )

            if self.transform:
                data = self.transform(data)

            processed_data.append(data)
        return processed_data

    def setup(self, stage: Optional[str] = None):
        """Prepares data by fitting scalers and determining padding size."""
        if (stage == "fit" or stage is None) and self.train_raw:
            if self.max_num_places is None:
                self.max_num_places = max(net.spn.shape[0] for net in self.train_raw)
                print(f"Determined max number of places for padding: {self.max_num_places}")

            # Fit the label scaler
            if self.label_scaler is None:
                print("Fitting label scaler on training data...")
                labels_to_fit = [net.get_analysis_result(self.hparams.label_to_predict) for net in self.train_raw]
                train_labels = np.array(labels_to_fit).reshape(-1, 1)
                self.label_scaler = StandardScaler()
                self.label_scaler.fit(train_labels)

            # Process the datasets
            self.train_data = self._process_list(self.train_raw, "Processing training data")
            if self.val_raw:
                self.val_data = self._process_list(self.val_raw, "Processing validation data")

        if (stage == "test" or stage is None) and self.test_raw:
            # Ensure max_num_places is set (e.g., if only running test)
            if self.max_num_places is None:
                raise RuntimeError(
                    "`max_num_places` is not set. Please run setup('fit') first or ensure a training set is provided."
                )
            self.test_data = self._process_list(self.test_raw, "Processing test data")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_data:
            return None
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_data:
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
