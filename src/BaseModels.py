"""
This module defines the base Lightning module for all GNN models in this project.
"""

from typing import Dict, Any, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import ModuleDict
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
    instantiation, optimizer configuration, and the training/validation/test steps.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        metrics_config: Dict[str, List[str]] = None,
    ):
        """
        Initializes the BaseGNNModule.

        Args:
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
            metrics_config: A dictionary specifying which metrics to use for each split.
        """
        super().__init__()
        self.save_hyperparameters("learning_rate", "weight_decay")

        # Use a default config if none is provided
        self.metrics_config = metrics_config or {
            "train": ["mae"],
            "val": ["mae", "rmse", "r2", "medae"],
            "test": ["mae", "rmse", "r2", "mape", "medae", "explainedvariance", "smape", "maxerror"],
        }
        self._initialize_metrics()

    def _get_metric_class(self, metric_name: str) -> Any:
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
        name_lower = metric_name.lower()
        if name_lower not in metric_map:
            raise ValueError(f"Unsupported metric: {metric_name}")
        return metric_map[name_lower]()

    def _initialize_metrics(self):
        """Initializes the metrics for all splits and registers them."""
        self.metrics = ModuleDict()
        for split in self.metrics_config:
            metrics_to_add = {name: self._get_metric_class(name) for name in self.metrics_config[split]}
            self.metrics[split] = MetricCollection(metrics_to_add)

    def forward(self, batch: Data | HeteroData) -> Any:
        """The forward pass of the model."""
        raise NotImplementedError("Subclasses must implement the `forward` method.")

    def _calculate_loss(self, preds: Any, targets: Any) -> torch.Tensor:
        """Calculates the loss for a batch."""
        raise NotImplementedError("Subclasses must implement the `_calculate_loss` method.")

    def training_step(self, batch: Data | HeteroData, batch_idx: int) -> torch.Tensor:
        """The training step for the model."""
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["train"].update(preds, targets)
        return loss

    def on_train_epoch_end(self):
        """Logs computed training metrics at the end of the epoch."""
        computed_metrics = self.metrics["train"].compute()
        self.log_dict({f"train/{k}": v for k, v in computed_metrics.items()})
        self.metrics["train"].reset()

    def validation_step(self, batch: Data | HeteroData, batch_idx: int):
        """The validation step for the model."""
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["val"].update(preds, targets)

    def on_validation_epoch_end(self):
        """Logs computed validation metrics at the end of the epoch."""
        computed_metrics = self.metrics["val"].compute()
        self.log_dict({f"val/{k}": v for k, v in computed_metrics.items()})
        self.metrics["val"].reset()

    def test_step(self, batch: Data | HeteroData, batch_idx: int):
        """The test step for the model."""
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.metrics["test"].update(preds, targets)

    def on_test_epoch_end(self):
        """Logs computed test metrics at the end of the epoch."""
        computed_metrics = self.metrics["test"].compute()
        self.log_dict({f"test/{k}": v for k, v in computed_metrics.items()})
        self.metrics["test"].reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the AdamW optimizer for the model."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
