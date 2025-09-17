"""
This module provides PyTorch Lightning implementations of Heterogeneous Graph Neural
Networks for evaluating Stochastic Petri Nets (SPNs).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, ModuleList
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, Linear, RGATConv, HEATConv
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


class LightningSPNModule(BaseGNNModule, ABC):
    """An abstract base class for heterogeneous GNN models for SPNs.

    This class extends the BaseGNNModule to handle the specific requirements
    of heterogeneous graphs, including per-node-type metric tracking, by
    overriding the metric-related hooks.
    """

    def __init__(self, metrics_config: Dict[str, List[str]] = None, **kwargs: Any):
        # This will be set by the subclass, but we provide a default
        self.node_types = ["place", "transition"]
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
        """Initializes both aggregate and per-node-type metrics."""
        # Initialize aggregate metrics
        self.metrics = ModuleDict()
        for split in self.metrics_config:
            metrics_to_add = {name: self._get_metric_class(name) for name in self.metrics_config[split]}
            self.metrics[split] = MetricCollection(metrics_to_add)

        # Initialize per-node-type metrics
        self.per_node_type_metrics = ModuleDict()
        for split in self.metrics_config:
            metrics_to_add = {
                f"{nt}_{name}": self._get_metric_class(name)
                for nt in self.node_types
                for name in self.metrics_config[split]
            }
            self.per_node_type_metrics[split] = MetricCollection(metrics_to_add)

    def _calculate_loss(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculates the total Mean Squared Error loss across all node types."""
        total_loss = 0.0
        for node_type, y_pred in preds.items():
            y_true = targets.get(node_type)
            if y_true is not None:
                total_loss += F.mse_loss(y_pred.squeeze(), y_true.float())
        return total_loss if total_loss > 0.0 else torch.tensor(0.0, device=self.device)

    def _update_metrics(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], split: str
    ):
        """Overrides the base method to update both aggregate and per-node-type metrics."""
        # Update aggregate metrics
        agg_preds = torch.cat([p.squeeze() for p in preds.values() if p.numel() > 0])
        agg_targets = torch.cat([t for t in targets.values() if t.numel() > 0])
        if agg_preds.numel() > 0:
            super()._update_metrics(agg_preds, agg_targets, split)

        # Update per-node-type metrics
        for node_type, y_pred in preds.items():
            y_true = targets.get(node_type)
            if y_true is not None and y_pred.numel() > 0:
                for metric_name in self.metrics_config[split]:
                    metric_key = f"{node_type}_{metric_name}"
                    self.per_node_type_metrics[split][metric_key].update(y_pred.squeeze(), y_true)

    def _log_and_reset_metrics(self, split: str):
        """Overrides the base method to log and reset both sets of metrics."""
        # Log and reset aggregate metrics
        super()._log_and_reset_metrics(split)

        # Log and reset per-node-type metrics
        computed_per_node = self.per_node_type_metrics[split].compute()
        self.log_dict({f"{split}/{k}": v for k, v in computed_per_node.items()})
        self.per_node_type_metrics[split].reset()


class BaseHeteroGNN(LightningSPNModule):
    """An abstract base class for specific heterogeneous GNN model implementations."""

    def __init__(self, out_channels: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters("out_channels")
        self.convs = ModuleList()
        self.lin = ModuleDict({nt: Linear(-1, out_channels) for nt in self.node_types})

    @abstractmethod
    def _create_conv_layers(self) -> None:
        """Abstract method to create the GNN convolution layers."""
        pass

    def forward(self, batch: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """The forward pass for the heterogeneous GNN model."""
        x_dict = batch.x_dict
        for conv in self.convs:
            extra_args = {"node_type_dict": batch.node_type_dict} if "node_type_dict" in batch else {}
            x_dict = conv(x_dict, batch.edge_index_dict, edge_attr_dict=batch.edge_attr_dict, **extra_args)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        preds_dict = {nt: self.lin[nt](x) for nt, x in x_dict.items()}
        targets_dict = {nt: batch[nt].y for nt in self.node_types if hasattr(batch[nt], "y")}

        return preds_dict, targets_dict


class RGAT_SPN_Model(BaseHeteroGNN):
    """An SPN evaluation model using Relational Graph Attention (RGAT) layers."""

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        edge_dim: int,
        **kwargs: Any,
    ):
        super().__init__(out_channels=out_channels, **kwargs)
        self.save_hyperparameters()
        self._create_conv_layers()

    def _create_conv_layers(self) -> None:
        """Creates the stack of RGAT convolutional layers."""
        in_channels = self.hparams.in_channels_dict
        out_dim = self.hparams.hidden_channels
        num_heads = self.hparams.num_heads
        edge_dim = self.hparams.edge_dim

        for i in range(self.hparams.num_layers):
            conv_dict = {
                rel: RGATConv(
                    in_channels=(in_channels[rel[0]], in_channels[rel[2]]),
                    out_channels=out_dim,
                    num_heads=num_heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
                for rel in [("place", "to", "transition"), ("transition", "to", "place")]
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            in_channels = {key: out_dim * num_heads for key in in_channels.keys()}

        self.lin = ModuleDict({nt: Linear(in_channels[nt], self.hparams.out_channels) for nt in self.node_types})


class HEAT_SPN_Model(BaseHeteroGNN):
    """An SPN evaluation model using Heterogeneous Edge-Attention Transformer (HEAT) layers."""

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_heads: int,
        edge_dim: int,
        num_node_types: int,
        num_edge_types: int,
        node_type_emb_dim: int,
        edge_type_emb_dim: int,
        **kwargs: Any,
    ):
        super().__init__(out_channels=out_channels, **kwargs)
        self.save_hyperparameters()
        self._create_conv_layers()

    def _create_conv_layers(self) -> None:
        """Creates the stack of HEAT convolutional layers."""
        in_channels = self.hparams.in_channels_dict
        out_dim = self.hparams.hidden_channels

        for i in range(self.hparams.num_layers):
            conv_dict = {
                rel: HEATConv(
                    in_channels=(in_channels[rel[0]], in_channels[rel[2]]),
                    out_channels=out_dim,
                    num_heads=self.hparams.num_heads,
                    edge_dim=self.hparams.edge_dim,
                    num_node_types=self.hparams.num_node_types,
                    num_edge_types=self.hparams.num_edge_types,
                    node_type_emb_dim=self.hparams.node_type_emb_dim,
                    edge_type_emb_dim=self.hparams.edge_type_emb_dim,
                )
                for rel in [("place", "to", "transition"), ("transition", "to", "place")]
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            in_channels = {key: out_dim for key in in_channels.keys()}

        self.lin = ModuleDict({nt: Linear(in_channels[nt], self.hparams.out_channels) for nt in self.node_types})
