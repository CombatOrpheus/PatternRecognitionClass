"""
This module provides a flexible PyTorch Lightning implementation of a homogeneous
Graph Neural Network using GCNConv layers for evaluating Stochastic Petri Nets.

This version includes robust metric tracking for regression tasks.
"""
from typing import Dict, Any, Literal

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class FlexibleGNN_SPN_Model(pl.LightningModule):
    """
    A flexible homogeneous GNN for SPN evaluation using GCNConv layers.

    This model can be configured to perform either graph-level or node-level
    predictions and includes comprehensive metric tracking (MAE, MSE, RMSE, R2).
    """

    def __init__(
            self,
            node_features_dim: int,
            hidden_dim: int,
            out_channels: int,
            num_layers: int,
            prediction_level: Literal['graph', 'node'] = 'graph',
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
    ):
        """
        Args:
            node_features_dim (int): Dimensionality of the input node features (e.g., 4).
            hidden_dim (int): Dimensionality of the hidden GCN layers.
            out_channels (int): Dimensionality of the final output. For single value
                                prediction, this is 1.
            num_layers (int): The number of GCNConv layers.
            prediction_level (Literal['graph', 'node']): The prediction task type.
                                'graph' for a single prediction per graph.
                                'node' for a prediction for each node.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(node_features_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.output_mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, out_channels)
        )

        # --- Instantiate metrics for each data split (train/val/test) ---
        # This is best practice to ensure metric states are kept separate.
        for split in ['train', 'val', 'test']:
            setattr(self, f"{split}_mae", MeanAbsoluteError())
            setattr(self, f"{split}_mse", MeanSquaredError())
            setattr(self, f"{split}_rmse", MeanSquaredError(squared=False))
            setattr(self, f"{split}_r2", R2Score())

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Defines the computation performed at every call.
        """
        x, edge_index, edge_attr, batch_map = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        edge_weight = edge_attr.squeeze(-1) if edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        if self.hparams.prediction_level == 'graph':
            embedding = global_add_pool(x, batch_map)
        else:  # 'node' level
            embedding = x

        prediction = self.output_mlp(embedding)

        if self.hparams.out_channels == 1:
            return prediction.squeeze(-1)
        return prediction

    def _common_step(self, batch: Data, batch_idx: int, prefix: str) -> torch.Tensor:
        """Shared logic for training, validation, and testing."""
        # Get raw predictions from the model.
        # For 'graph' level, this is one prediction per graph.
        # For 'node' level, this is one prediction per node.
        y_pred_raw = self(batch)
        y_true = batch.y

        # For node-level tasks, filter predictions to match the ground truth.
        # The ground truth `y` may only correspond to a subset of nodes
        # (e.g., only places or only transitions). We use a mask derived
        # from the node features to select the correct predictions.
        if self.hparams.prediction_level == 'node':
            # The first column of `batch.x` is the 'is_place' feature (1.0 or 0.0).
            # The second column is the 'is_transition' feature.
            # We convert these to boolean masks.
            place_mask = batch.x[:, 0].bool()
            transition_mask = batch.x[:, 1].bool()

            # We determine which mask to use by checking which one's count
            # of `True` values matches the number of labels in y_true.
            if place_mask.sum() == y_true.size(0):
                y_pred = y_pred_raw[place_mask]
            elif transition_mask.sum() == y_true.size(0):
                y_pred = y_pred_raw[transition_mask]
            else:
                # If no mask matches, it indicates a data preparation issue.
                # Raise a descriptive error to help with debugging.
                raise ValueError(
                    f"Shape mismatch in node-level prediction: "
                    f"Model produced {y_pred_raw.size(0)} outputs for all nodes, "
                    f"but ground truth has {y_true.size(0)} labels. "
                    f"The number of places ({place_mask.sum()}) and "
                    f"transitions ({transition_mask.sum()}) in the batch "
                    f"do not match the label count. Check your `SPNDataModule` "
                    f"and the `label_to_predict` setting."
                )
        else:  # 'graph' level
            y_pred = y_pred_raw

        # Ensure y_true is float for loss calculation and metrics
        y_true = y_true.float()

        # Final sanity check for shape consistency before calculating loss
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Final prediction shape {y_pred.shape} does not match "
                f"ground truth shape {y_true.shape}. This can happen if "
                f"masking logic is incorrect or data is malformed."
            )

        loss = F.mse_loss(y_pred, y_true)
        self.log(
            f"{prefix}_loss", loss, on_step=False, on_epoch=True,
            prog_bar=True, batch_size=batch.num_graphs
        )

        # --- Update and log all metrics for the current step ---
        mae = getattr(self, f"{prefix}_mae")
        mse = getattr(self, f"{prefix}_mse")
        rmse = getattr(self, f"{prefix}_rmse")
        r2 = getattr(self, f"{prefix}_r2")

        # Update metrics with the current batch's predictions and labels
        mae.update(y_pred, y_true)
        mse.update(y_pred, y_true)
        rmse.update(y_pred, y_true)
        r2.update(y_pred, y_true)

        # Log the metrics. `on_epoch=True` tells Lightning to aggregate them
        # over the entire epoch before printing the final value.
        self.log(f"{prefix}_mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{prefix}_r2", r2, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}
