"""This module provides a flexible and modular PyTorch Lightning implementation of
homogeneous Graph Neural Networks for evaluating Stochastic Petri Nets (SPNs).
"""

from typing import Any, Dict, List, Literal, Tuple

import torch
import torch.nn.functional as F
from torch.nn import GINConv, GATConv, Linear, ReLU, Sequential, ModuleDict
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
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    MeanAbsolutePercentageError,
    ExplainedVariance,
    SymmetricMeanAbsolutePercentageError,
)

from src.BaseModels import BaseGNNModule
from src.CustomMetrics import MaxError, MedianAbsoluteError


class BaseGNN_SPN_Model(BaseGNNModule):
    """A base class for Homogeneous GNN models.

    This class provides shared utilities like the GNN layer factory, the
    common loss calculation method, and the metric initialization logic.
    """

    def __init__(self, metrics_config: Dict[str, List[str]] = None, **kwargs: Any):
        self.metrics_config = metrics_config or {
            "train": ["mae"],
            "val": ["mae", "rmse", "r2", "medae"],
            "test": ["mae", "rmse", "r2", "mape", "medae", "explainedvariance", "smape", "maxerror"],
        }
        super().__init__(**kwargs)

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

    def _calculate_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Mean Squared Error loss."""
        return F.mse_loss(preds, targets)


class GraphGNN_SPN_Model(BaseGNN_SPN_Model):
    """A GNN model for graph-level prediction tasks on SPNs."""

    def __init__(
        self,
        node_features_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        gnn_operator_name: Literal["gcn", "cheb", "tag", "sgc", "ssg"] = "gcn",
        num_layers_mlp: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
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

    def forward(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the model."""
        x, edge_index, edge_attr, batch_map = (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        embedding = global_add_pool(x, batch_map)
        preds = self.output_mlp(embedding)
        preds = preds.squeeze(-1) if self.hparams.out_channels == 1 else preds

        targets = batch.y.float()
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, targets={targets.shape}")

        return preds, targets


class NodeGNN_SPN_Model(BaseGNN_SPN_Model):
    """A GNN model for node-level prediction tasks on SPNs."""

    def __init__(
        self,
        node_features_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        gnn_operator_name: Literal["gcn", "cheb", "tag", "sgc", "ssg"] = "gcn",
        num_layers_mlp: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
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

    def forward(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the model."""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        preds_raw = self.output_mlp(x)
        preds_raw = preds_raw.squeeze(-1) if self.hparams.out_channels == 1 else preds_raw

        # --- Masking logic for node-level predictions ---
        targets = batch.y.float()
        place_mask = batch.x[:, 0].bool()
        transition_mask = batch.x[:, 1].bool()

        if place_mask.sum() == targets.size(0):
            preds = preds_raw[place_mask]
        elif transition_mask.sum() == targets.size(0):
            preds = preds_raw[transition_mask]
        else:
            raise ValueError(
                f"Node-level prediction shape mismatch: Ground truth labels ({targets.size(0)}) "
                f"does not match places ({place_mask.sum()}) or transitions ({transition_mask.sum()})."
            )

        if preds.shape != targets.shape:
            raise ValueError(f"Final shape mismatch: preds={preds.shape}, targets={targets.shape}")

        return preds, targets


class MixedGNN_SPN_Model(BaseGNN_SPN_Model):
    """A GNN model with a predefined sequence of GAT, GCN, and GIN layers."""

    def __init__(
        self,
        node_features_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers_mlp: int = 2,
        heads: int = 4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.conv1 = GATConv(node_features_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GCNConv(hidden_dim * heads, hidden_dim)
        gin_mlp = Sequential(Linear(hidden_dim, hidden_dim * 2), ReLU(), Linear(hidden_dim * 2, hidden_dim))
        self.conv3 = GINConv(gin_mlp)
        self.output_mlp = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=out_channels,
            num_layers=num_layers_mlp,
            act="relu",
        )

    def forward(self, batch: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass of the model."""
        x, edge_index, batch_map = batch.x, batch.edge_index, batch.batch

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.leaky_relu(self.conv3(x, edge_index))

        embedding = global_add_pool(x, batch_map)
        preds = self.output_mlp(embedding)
        preds = preds.squeeze(-1) if self.hparams.out_channels == 1 else preds

        targets = batch.y.float()
        if preds.shape != targets.shape:
            raise ValueError(f"Shape mismatch: preds={preds.shape}, targets={targets.shape}")

        return preds, targets
