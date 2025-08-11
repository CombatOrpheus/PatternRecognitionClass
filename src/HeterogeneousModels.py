"""
This module provides PyTorch Lightning implementations of Heterogeneous Graph Neural
Networks for evaluating Stochastic Petri Nets (SPNs).

It includes two main models:
1.  RGAT_SPN_Model: Utilizes Relational Graph Attention (RGAT) layers.
2.  HEAT_SPN_Model: Utilizes Heterogeneous Edge-based Attention Transformer (HEAT) layers.

Both models are built on a common base class, `LightningSPNModule`, which
handles the training, validation, and optimization logic with comprehensive metric tracking.
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.nn import ModuleDict
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, RGATConv, HEATConv, Linear
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class LightningSPNModule(pl.LightningModule):
    """
    A base PyTorch Lightning module for SPN models.

    This class encapsulates the shared logic for training, validation, testing,
    and optimizer configuration. Subclasses are expected to implement the
    `__init__` and `forward` methods to define the specific GNN architecture.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """
        Initializes the base Lightning module.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        """
        super().__init__()
        # `save_hyperparameters` will be called in the child classes to ensure
        # all their specific hparams are also saved. We store these for the optimizer.
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # --- Instantiate metrics for each node type and data split ---
        self.node_types = ["place", "transition"]
        for split in ["train", "val", "test"]:
            metrics = ModuleDict(
                {
                    "mae": ModuleDict({nt: MeanAbsoluteError() for nt in self.node_types}),
                    "mse": ModuleDict({nt: MeanSquaredError() for nt in self.node_types}),
                    "rmse": ModuleDict({nt: MeanSquaredError(squared=False) for nt in self.node_types}),
                    "r2": ModuleDict({nt: R2Score() for nt in self.node_types}),
                }
            )
            setattr(self, f"{split}_metrics", metrics)

    def _common_step(self, batch: HeteroData, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        Performs a common step for training, validation, and testing.

        This method runs the forward pass, calculates the loss, and updates
        metrics for each node type with a corresponding label.
        """
        output_dict = self(batch)
        total_loss = 0.0
        metrics = getattr(self, f"{prefix}_metrics")

        # Calculate loss and update metrics for each node type that has a ground truth label
        for node_type, y_pred in output_dict.items():
            if node_type in batch.y_dict:
                y_true = batch.y_dict[node_type]
                loss = F.mse_loss(y_pred.squeeze(), y_true.float())
                total_loss += loss

                # Update metrics for the specific node type
                for metric_name, metric_module in metrics.items():
                    metric = metric_module[node_type]
                    metric.update(y_pred.squeeze(), y_true)
                    # Log with a hierarchical structure for better organization
                    self.log(
                        f"{prefix}/{node_type}/{metric_name}",
                        metric,
                        on_step=False,
                        on_epoch=True,
                    )

        if total_loss == 0.0:
            # This can happen if a batch contains no labeled nodes.
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Log the aggregated total loss
        self.log(f"{prefix}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the AdamW optimizer for training.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return {"optimizer": optimizer}

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """
        The forward pass of the model. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class RGAT_SPN_Model(LightningSPNModule):
    """
    An SPN evaluation model using Relational Graph Attention (RGAT) layers.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        edge_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = -1 if i == 0 else hidden_channels * num_heads
            conv = HeteroConv(
                {
                    rel: RGATConv(
                        in_channels=in_c,
                        out_channels=hidden_channels,
                        num_heads=num_heads,
                        edge_dim=edge_dim,
                        add_self_loops=False,
                    )
                    for rel in [
                        ("place", "to", "transition"),
                        ("transition", "to", "place"),
                    ]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = torch.nn.ModuleDict(
            {
                "place": Linear(hidden_channels * num_heads, out_channels),
                "transition": Linear(hidden_channels * num_heads, out_channels),
            }
        )

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        # *** BUG FIX APPLIED HERE ***
        output_dict = {node_type: self.linnode_type for node_type, x in x_dict.items()}

        return output_dict


class HEAT_SPN_Model(LightningSPNModule):
    """
    An SPN evaluation model using Heterogeneous Edge-based Attention Transformer (HEAT) layers.
    """

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        edge_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = -1 if i == 0 else hidden_channels
            conv = HeteroConv(
                {
                    rel: HEATConv(
                        in_channels=in_c,
                        out_channels=hidden_channels,
                        num_heads=num_heads,
                        edge_dim=edge_dim,
                    )
                    for rel in [
                        ("place", "to", "transition"),
                        ("transition", "to", "place"),
                    ]
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.lin = torch.nn.ModuleDict(
            {
                "place": Linear(hidden_channels, out_channels),
                "transition": Linear(hidden_channels, out_channels),
            }
        )

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        node_type_dict = batch.node_type_dict

        for conv in self.convs:
            x_dict = conv(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
                node_type_dict=node_type_dict,
            )
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        output_dict = {node_type: self.linnode_type for node_type, x in x_dict.items()}

        return output_dict
