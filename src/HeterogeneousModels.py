"""
This module provides PyTorch Lightning implementations of Heterogeneous Graph Neural
Networks for evaluating Stochastic Petri Nets (SPNs).

It includes two main models:
1.  RGAT_SPN_Model: Utilizes Relational Graph Attention (RGAT) layers.
2.  HEAT_SPN_Model: Utilizes Heterogeneous Edge-based Attention Transformer (HEAT) layers.

Both models are built on a common base class, `LightningSPNModule`, which
handles the training, validation, and optimization logic with comprehensive metric tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, ModuleList
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGATConv, HEATConv, Linear
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class LightningSPNModule(pl.LightningModule):
    """
    A base PyTorch Lightning module for SPN models. Encapsulates shared logic
    for training, validation, testing, and optimizer configuration.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.node_types = ["place", "transition"]
        for split in ["train", "val", "test"]:
            metrics_per_node = ModuleDict(
                {
                    nt: MetricCollection(
                        {
                            "mae": MeanAbsoluteError(),
                            "mse": MeanSquaredError(),
                            "rmse": MeanSquaredError(squared=False),
                            "r2": R2Score(),
                        }
                    )
                    for nt in self.node_types
                }
            )
            setattr(self, f"{split}_metrics", metrics_per_node)

    def _common_step(self, batch: HeteroData, prefix: str) -> torch.Tensor:
        output_dict = self(batch)
        total_loss = 0.0
        metrics_dict = getattr(self, f"{prefix}_metrics")

        for node_type, y_pred in output_dict.items():
            y_true = batch[node_type].y if hasattr(batch[node_type], "y") else None
            if y_true is None:
                continue

            loss = F.mse_loss(y_pred.squeeze(), y_true.float())
            total_loss += loss

            metrics = metrics_dict[node_type]
            metrics.update(y_pred.squeeze(), y_true)
            self.log_dict({f"{prefix}/{node_type}/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True)

        if total_loss == 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log(f"{prefix}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        return {
            "optimizer": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        }

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")


class BaseHeteroGNN(LightningSPNModule, ABC):
    """
    An abstract base class for heterogeneous GNN models.
    It implements the shared forward pass logic.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs.get("learning_rate", 1e-3), kwargs.get("weight_decay", 1e-5))
        self.save_hyperparameters()
        self.convs = ModuleList()
        # The final linear layers are now defined in the child classes
        self.lin = ModuleDict()

    @abstractmethod
    def _create_conv_layers(self) -> None:
        """Must be implemented by subclasses to build the GNN layer stack."""
        pass

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = batch.x_dict
        for conv in self.convs:
            extra_args = {}
            if "node_type_dict" in batch:
                extra_args["node_type_dict"] = batch.node_type_dict

            x_dict = conv(x_dict, batch.edge_index_dict, edge_attr_dict=batch.edge_attr_dict, **extra_args)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        return {nt: self.lin[nt](x) for nt, x in x_dict.items()}


class RGAT_SPN_Model(BaseHeteroGNN):
    """
    An SPN evaluation model using Relational Graph Attention (RGAT) layers.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._create_conv_layers()

    def _create_conv_layers(self) -> None:
        in_channels = self.hparams.in_channels_dict
        # The output dimension of an RGAT layer is hidden_channels * num_heads
        layer_output_dim = self.hparams.hidden_channels * self.hparams.num_heads

        for i in range(self.hparams.num_layers):
            conv = HeteroConv(
                {
                    rel: RGATConv(
                        in_channels=(in_channels[rel[0]], in_channels[rel[2]]),
                        out_channels=self.hparams.hidden_channels,
                        num_heads=self.hparams.num_heads,
                        edge_dim=self.hparams.edge_dim,
                        add_self_loops=False,
                    )
                    for rel in [("place", "to", "transition"), ("transition", "to", "place")]
                },
                aggr="sum",
            )
            self.convs.append(conv)
            # Update in_channels for the next layer
            in_channels = {key: layer_output_dim for key in in_channels.keys()}

        # Define the final linear layers with the correct input dimension
        self.lin = ModuleDict({nt: Linear(layer_output_dim, self.hparams.out_channels) for nt in self.node_types})


class HEAT_SPN_Model(BaseHeteroGNN):
    """
    An SPN evaluation model using Heterogeneous Edge-based Attention Transformer (HEAT) layers.
    """

    def __init__(
        self, in_channels_dict: Dict[str, int], hidden_channels: int, out_channels: int, num_layers: int, **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._create_conv_layers()

    def _create_conv_layers(self) -> None:
        in_channels = self.hparams.in_channels_dict
        # The output dimension of a HEAT layer is just hidden_channels
        layer_output_dim = self.hparams.hidden_channels

        for i in range(self.hparams.num_layers):
            conv = HeteroConv(
                {
                    rel: HEATConv(
                        in_channels=(in_channels[rel[0]], in_channels[rel[2]]),
                        out_channels=self.hparams.hidden_channels,
                        num_heads=self.hparams.num_heads,
                        edge_dim=self.hparams.edge_dim,
                        num_node_types=self.hparams.num_node_types,
                        num_edge_types=self.hparams.num_edge_types,
                        node_type_emb_dim=self.hparams.node_type_emb_dim,
                        edge_type_emb_dim=self.hparams.edge_type_emb_dim,
                    )
                    for rel in [("place", "to", "transition"), ("transition", "to", "place")]
                },
                aggr="sum",
            )
            self.convs.append(conv)
            # Update in_channels for the next layer
            in_channels = {key: layer_output_dim for key in in_channels.keys()}

        # Define the final linear layers with the correct input dimension
        self.lin = ModuleDict({nt: Linear(layer_output_dim, self.hparams.out_channels) for nt in self.node_types})
