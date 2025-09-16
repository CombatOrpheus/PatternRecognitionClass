"""
This module defines the base Lightning module for all GNN models in this project.
"""

from typing import Dict, Any, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
    ExplainedVariance,
    SymmetricMeanAbsolutePercentageError,
)
from src.CustomMetrics import MaxError, MedianAbsoluteError

class BaseGNNModule(pl.LightningModule):
    """
    A base class for GNN models that handles common functionality like metric
    instantiation, optimizer configuration, and the basic training, validation,
    and test step logic.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        metrics: Dict[str, List[str]] = None,
    ):
        """
        Initializes the BaseGNNModule.

        Args:
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            metrics: A dictionary specifying which metrics to use for each split.
        """
        super().__init__()
        self.save_hyperparameters("learning_rate", "weight_decay")
        self.metrics_config = metrics or {
            "train": ["mae"],
            "val": ["mae", "rmse", "r2", "medae"],
            "test": ["mae", "rmse", "r2", "mape", "medae", "explainedvariance", "smape", "maxerror"],
        }
        self._initialize_metrics()

    def _get_metric_class(self, metric_name: str):
        """Returns the metric class for a given metric name."""
        metric_map = {
            "mae": MeanAbsoluteError,
            "mse": MeanSquaredError,
            "rmse": lambda: MeanSquaredError(squared=False),
            "r2": R2Score,
            "mape": MeanAbsolutePercentageError,
            "medae": MedianAbsoluteError,
            "explainedvariance": ExplainedVariance,
            "smape": SymmetricMeanAbsolutePercentageError,
            "maxerror": MaxError,
        }
        if metric_name.lower() not in metric_map:
            raise ValueError(f"Unsupported metric: {metric_name}")
        return metric_map[metric_name.lower()]

    def _initialize_metrics(self):
        """Instantiates regression metrics using MetricCollection for each data split."""
        for split, metric_names in self.metrics_config.items():
            collection = MetricCollection(
                {name: self._get_metric_class(name)() for name in metric_names}
            )
            setattr(self, f"{split}_metrics", collection)

    def training_step(self, batch: Data | HeteroData, batch_idx: int) -> torch.Tensor:
        """The training step for the model."""
        return self._common_step(batch, "train")

    def validation_step(self, batch: Data | HeteroData, batch_idx: int) -> torch.Tensor:
        """The validation step for the model."""
        return self._common_step(batch, "val")

    def test_step(self, batch: Data | HeteroData, batch_idx: int) -> torch.Tensor:
        """The test step for the model."""
        return self._common_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the AdamW optimizer for the model."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}

    def _common_step(self, batch: Data | HeteroData, prefix: str) -> torch.Tensor:
        """A common step for training, validation, and testing."""
        raise NotImplementedError("Subclasses must implement the `_common_step` method.")
