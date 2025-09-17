"""
This module defines the base Lightning module for all GNN models in this project.
It centralizes the training, validation, and testing loops, and delegates
specific logic like metric updates to subclasses.
"""

from typing import Any, Dict

import lightning.pytorch as pl
import torch


class BaseGNNModule(pl.LightningModule):
    """
    A base class for all GNN models that handles the core training, validation,
    and testing loops.

    Subclasses are expected to implement:
    - `forward`: To return predictions and targets.
    - `_calculate_loss`: To compute the loss.
    - `_initialize_metrics`: To set up the `self.metrics` ModuleDict.
    And optionally override:
    - `_update_metrics`: To handle complex metric state updates (e.g., for heterogeneous data).
    - `_log_and_reset_metrics`: To handle complex metric logging.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters("learning_rate", "weight_decay")
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Initializes the `self.metrics` ModuleDict."""
        raise NotImplementedError("Subclasses must implement `_initialize_metrics`.")

    def forward(self, batch: Any) -> Any:
        """The forward pass of the model."""
        raise NotImplementedError("Subclasses must implement the `forward` method.")

    def _calculate_loss(self, preds: Any, targets: Any) -> torch.Tensor:
        """Calculates the loss for a batch."""
        raise NotImplementedError("Subclasses must implement the `_calculate_loss` method.")

    def _update_metrics(self, preds: torch.Tensor, targets: torch.Tensor, split: str):
        """Updates the metric collection for a given split."""
        self.metrics[split].update(preds, targets)

    def _log_and_reset_metrics(self, split: str):
        """Computes, logs, and resets the metrics for a given split."""
        computed_metrics = self.metrics[split].compute()
        self.log_dict({f"{split}/{k}": v for k, v in computed_metrics.items()})
        self.metrics[split].reset()

    # --- Core Training, Validation, and Test Loops ---
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "train")
        return loss

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")

    def validation_step(self, batch: Any, batch_idx: int):
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "val")

    def on_validation_epoch_end(self):
        self._log_and_reset_metrics("val")

    def test_step(self, batch: Any, batch_idx: int):
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "test")

    def on_test_epoch_end(self):
        self._log_and_reset_metrics("test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the AdamW optimizer for the model."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
