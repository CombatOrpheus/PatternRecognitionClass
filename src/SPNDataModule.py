"""
This module defines the SPNDataModule, a PyTorch Lightning DataModule for
handling and preparing Stochastic Petri Net (SPN) data for GNN models.

This enhanced version includes:
- Support for applying torch_geometric transforms.
- Automatic normalization of regression labels for improved training stability.
"""

from typing import Optional, List, Callable

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.PetriNets import SPNData, SPNAnalysisResultLabel


class SPNDataModule(pl.LightningDataModule):
    """
    A LightningDataModule for creating training, validation, and test datasets
    for Stochastic Petri Nets from pre-split data sources.
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
        """
        Args:
            label_to_predict (SPNAnalysisResultLabel): The analysis result to use as the label.
            train_data_list (List[SPNData]): Raw SPNData objects for training.
            val_data_list (Optional[List[SPNData]]): Optional list for validation.
            test_data_list (Optional[List[SPNData]]): Optional list for testing.
            batch_size (int): Batch size for the DataLoaders.
            num_workers (int): Number of subprocesses for data loading.
            transform (Optional[Callable]): A function/transform to apply to each Data object.
        """
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

    def _process_list(self, raw_data_list: List[SPNData], desc: Optional[str] = None) -> List[Data]:
        """Converts a list of raw SPNData objects to PyG Data objects."""
        processed_data = []
        for net in tqdm(raw_data_list, leave=False, desc=desc):
            node_features, edge_features, edge_pairs = net.to_information()
            label_raw = net.get_analysis_result(self.hparams.label_to_predict)

            # Ensure label is a NumPy array for consistent processing
            if not isinstance(label_raw, np.ndarray):
                label_raw = np.array([label_raw])

            # Normalize the label using the pre-fitted scaler
            if self.label_scaler:
                # Reshape to (N, 1) for the scaler, then transform
                label_scaled = self.label_scaler.transform(label_raw.reshape(-1, 1))
            else:
                # This case should ideally not be hit if setup is called correctly
                label_scaled = label_raw

            data = Data(
                x=from_numpy(node_features).float(),
                edge_attr=from_numpy(edge_features).float(),
                edge_index=from_numpy(edge_pairs).long(),
                y=from_numpy(label_scaled).float().flatten(),
            )

            # Apply the transform if it has been provided
            if self.transform:
                data = self.transform(data)

            processed_data.append(data)
        return processed_data

    def setup(self, stage: Optional[str] = None):
        """
        Prepares data. This method fits the label scaler, then processes the datasets.
        """
        # --- Fit the label scaler ONCE using only the training data ---
        if (stage == "fit" or stage is None) and self.train_raw and self.label_scaler is None:
            print("Fitting label scaler on training data...")

            # Aggregate all label values from the training set to fit the scaler
            labels_to_fit = []
            for net in self.train_raw:
                label = net.get_analysis_result(self.hparams.label_to_predict)
                if isinstance(label, np.ndarray):
                    labels_to_fit.extend(label.flatten())
                else:  # Handle scalar labels (float or int)
                    labels_to_fit.append(label)

            # Reshape for the scaler, which expects a 2D array
            train_labels = np.array(labels_to_fit).reshape(-1, 1)

            self.label_scaler = StandardScaler()
            self.label_scaler.fit(train_labels)

        # --- Process data splits ---
        if (stage == "fit" or stage is None) and self.train_raw:
            self.train_data = self._process_list(self.train_raw, "Processing training data...")
            if self.val_raw:
                self.val_data = self._process_list(self.val_raw, "Processing validation data...")

        if (stage == "test" or stage is None) and self.test_raw:
            self.test_data = self._process_list(self.test_raw, "Processing training data...")

    def inverse_transform_label(self, y_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a tensor of normalized model predictions back to their
        original scale.

        Args:
            y_tensor (torch.Tensor): The model's output tensor of predictions.

        Returns:
            np.ndarray: The predictions in their original, un-normalized scale.
        """
        if self.label_scaler is None:
            raise RuntimeError("Label scaler has not been fitted. Run setup('fit') first.")

        # Reshape tensor for the scaler, move to CPU, and convert to numpy
        y_reshaped = y_tensor.cpu().detach().numpy().reshape(-1, 1)
        return self.label_scaler.inverse_transform(y_reshaped)

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
            return None
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Returns the DataLoader for the test set."""
        if not self.test_data:
            return None
        return DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
