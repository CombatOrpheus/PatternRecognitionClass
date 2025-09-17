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

from src.BaseModels import BaseGNNModule


class LightningSPNModule(BaseGNNModule, ABC):
    """An abstract base class for heterogeneous GNN models for SPNs.

    This class extends the BaseGNNModule to handle the specific requirements
    of heterogeneous graphs, including per-node-type metric tracking.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.node_types = ["place", "transition"]
        self._initialize_per_node_type_metrics()

    def _initialize_per_node_type_metrics(self):
        """Initializes metrics for each node type and registers them."""
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

    def _update_metrics(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], split: str):
        """Updates both aggregate and per-node-type metrics."""
        # Update aggregate metrics
        agg_preds = torch.cat([p.squeeze() for p in preds.values()])
        agg_targets = torch.cat([t for t in targets.values()])
        self.metrics[split].update(agg_preds, agg_targets)

        # Update per-node-type metrics
        for node_type, y_pred in preds.items():
            y_true = targets.get(node_type)
            if y_true is not None:
                for metric_name in self.metrics_config[split]:
                    metric_key = f"{node_type}_{metric_name}"
                    self.per_node_type_metrics[split][metric_key].update(y_pred.squeeze(), y_true)

    def _log_and_reset_metrics(self, split: str):
        """Computes, logs, and resets both aggregate and per-node-type metrics."""
        # Aggregate metrics
        computed_metrics = self.metrics[split].compute()
        self.log_dict({f"{split}/{k}": v for k, v in computed_metrics.items()})
        self.metrics[split].reset()

        # Per-node-type metrics
        computed_per_node = self.per_node_type_metrics[split].compute()
        self.log_dict({f"{split}/{k}": v for k, v in computed_per_node.items()})
        self.per_node_type_metrics[split].reset()

    # --- Override training steps and hooks to handle per-node-type metrics ---
    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "train")
        return loss

    def on_train_epoch_end(self):
        self._log_and_reset_metrics("train")

    def validation_step(self, batch: HeteroData, batch_idx: int):
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "val")

    def on_validation_epoch_end(self):
        self._log_and_reset_metrics("val")

    def test_step(self, batch: HeteroData, batch_idx: int):
        preds, targets = self(batch)
        loss = self._calculate_loss(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._update_metrics(preds, targets, "test")

    def on_test_epoch_end(self):
        self._log_and_reset_metrics("test")


class BaseHeteroGNN(LightningSPNModule):
    """An abstract base class for specific heterogeneous GNN model implementations."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.convs = ModuleList()
        self.lin = ModuleDict({nt: Linear(-1, self.hparams.out_channels) for nt in self.node_types})

    @abstractmethod
    def _create_conv_layers(self) -> None:
        """Abstract method to create the GNN convolution layers."""
        pass

    def forward(self, batch: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """The forward pass for the heterogeneous GNN model."""
        x_dict = batch.x_dict
        for conv in self.convs:
            # Handle HEATConv's extra arguments if needed
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
            # Update input channels for the next layer
            in_channels = {key: out_dim * num_heads for key in in_channels.keys()}

        # Re-initialize final linear layers with correct input dimension
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
            # Update input channels for the next layer
            in_channels = {key: out_dim for key in in_channels.keys()}

        # Re-initialize final linear layers with correct input dimension
        self.lin = ModuleDict({nt: Linear(in_channels[nt], self.hparams.out_channels) for nt in self.node_types})
