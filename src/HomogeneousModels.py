"""
This module provides a flexible and modular PyTorch Lightning implementation of
homogeneous Graph Neural Networks for evaluating Stochastic Petri Nets (SPNs).

It features a base class for common logic and two specialized models for
graph-level and node-level regression tasks, each supporting multiple GNN
operator types (GCN, ChebConv, TAGConv, SGC, SSG).
"""

from typing import Dict, Any, Literal

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    TAGConv,
    SGConv,
    SSGConv,
    global_add_pool,
)
from torch_geometric.nn.models import MLP
from torchmetrics import MetricCollection, RelativeSquaredError
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
)


class BaseGNN_SPN_Model(pl.LightningModule):
    """
    A base class for GNN models that handles common functionality.
    This includes metric instantiation, optimizer configuration, and the
    basic training, validation, and test step logic.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters("learning_rate", "weight_decay")
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Instantiates regression metrics using MetricCollection."""
        for split in ["train", "val", "test"]:
            collection = MetricCollection(
                {
                    "mae": MeanAbsoluteError(),
                    "mse": MeanSquaredError(),
                    "rmse": MeanSquaredError(squared=False),
                    "rse": RelativeSquaredError(),
                    "rrse": RelativeSquaredError(squared=False),
                    "r2": R2Score(),
                    "mape": MeanAbsolutePercentageError(),
                }
            )
            setattr(self, f"{split}_metrics", collection)

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}

    def _common_step(self, batch: Data, prefix: str) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the `_common_step` method.")

    def _get_gnn_layer(self, name: str, in_dim: int, out_dim: int) -> torch.nn.Module:
        """Factory function to create a GNN layer based on its name."""
        name = name.lower()
        if name == "gcn":
            return GCNConv(in_dim, out_dim)
        elif name == "cheb":
            return ChebConv(in_dim, out_dim, K=self.hparams.gnn_k_hops)
        elif name == "tag":
            return TAGConv(in_dim, out_dim, K=self.hparams.gnn_k_hops)
        elif name == "sgc":
            return SGConv(in_dim, out_dim, K=self.hparams.gnn_k_hops)
        elif name == "ssg":
            return SSGConv(in_dim, out_dim, alpha=self.hparams.gnn_alpha, K=self.hparams.gnn_k_hops)
        else:
            raise ValueError(f"Unsupported GNN operator: {name}")


class GraphGNN_SPN_Model(BaseGNN_SPN_Model):
    """
    A GNN model for **graph-level** prediction tasks on SPNs.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        gnn_operator_name: Literal["gcn", "cheb", "tag", "sgc", "ssg"] = "gcn",
        num_layers_mlp: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gnn_k_hops: int = 3,
        gnn_alpha: float = 0.1,
    ):
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        self.convs.append(self._get_gnn_layer(gnn_operator_name, node_features_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(self._get_gnn_layer(gnn_operator_name, hidden_dim, hidden_dim))

        self.output_mlp = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers_mlp,
            act="relu",
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch_map = (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        embedding = global_add_pool(x, batch_map)
        prediction = self.output_mlp(embedding)

        return prediction.squeeze(-1) if self.hparams.out_channels == 1 else prediction

    def _common_step(self, batch: Data, prefix: str) -> torch.Tensor:
        y_pred = self(batch)
        y_true = batch.y.float()

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

        loss = F.mse_loss(y_pred, y_true)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        metrics = getattr(self, f"{prefix}_metrics")
        metrics.update(y_pred, y_true)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return loss


class NodeGNN_SPN_Model(BaseGNN_SPN_Model):
    """
    A GNN model for **node-level** prediction tasks on SPNs.
    """

    def __init__(
        self,
        node_features_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        gnn_operator_name: Literal["gcn", "cheb", "tag", "sgc", "ssg"] = "gcn",
        num_layers_mlp: int = 1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        gnn_k_hops: int = 3,
        gnn_alpha: float = 0.1,
    ):
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        self.convs.append(self._get_gnn_layer(gnn_operator_name, node_features_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(self._get_gnn_layer(gnn_operator_name, hidden_dim, hidden_dim))

        self.output_mlp = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers_mlp,
            act="relu",
        )

    def forward(self, batch: Data) -> torch.Tensor:
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        prediction = self.output_mlp(x)
        return prediction.squeeze(-1) if self.hparams.out_channels == 1 else prediction

    def _common_step(self, batch: Data, prefix: str) -> torch.Tensor:
        y_pred_raw = self(batch)
        y_true = batch.y.float()

        place_mask = batch.x[:, 0].bool()
        transition_mask = batch.x[:, 1].bool()

        if place_mask.sum() == y_true.size(0):
            y_pred = y_pred_raw[place_mask]
        elif transition_mask.sum() == y_true.size(0):
            y_pred = y_pred_raw[transition_mask]
        else:
            raise ValueError(
                "Node-level prediction shape mismatch: "
                f"Ground truth labels ({y_true.size(0)}) does not match "
                f"places ({place_mask.sum()}) or transitions ({transition_mask.sum()})."
            )

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Final shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

        loss = F.mse_loss(y_pred, y_true)
        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

        metrics = getattr(self, f"{prefix}_metrics")
        metrics.update(y_pred, y_true)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return loss
