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
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    ChebConv,
    TAGConv,
    SGConv,
    SSGConv,
    global_add_pool,
)
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score


class BaseGNN_SPN_Model(pl.LightningModule):
    """
    A base class for GNN models that handles common functionality.

    This includes metric instantiation, optimizer configuration, and the
    basic training, validation, and test step logic. It is designed to be
    inherited by specialized models for specific prediction tasks.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """
        Args:
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        """
        super().__init__()
        # Use save_hyperparameters to ensure these are stored in the checkpoint
        self.save_hyperparameters("learning_rate", "weight_decay")
        self._initialize_metrics()

    def _initialize_metrics(self):
        """Instantiates regression metrics for train, val, and test splits."""
        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_mae", MeanAbsoluteError())
            setattr(self, f"{split}_mse", MeanSquaredError())
            setattr(self, f"{split}_rmse", MeanSquaredError(squared=False))
            setattr(self, f"{split}_r2", R2Score())

    def _log_metrics(self, prefix: str, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Updates and logs all metrics for a given data split."""
        for metric_name in ["mae", "mse", "rmse", "r2"]:
            metric = getattr(self, f"{prefix}_{metric_name}")
            metric.update(y_pred, y_true)
            # Log with prog_bar for key metrics, and on_epoch for all
            prog_bar = metric_name in ["mae", "rmse"]
            self.log(
                f"{prefix}_{metric_name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
            )

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """Performs a single test step."""
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return {"optimizer": optimizer}

    def _common_step(self, batch: Data, batch_idx: int, prefix: str) -> torch.Tensor:
        """
        A placeholder for the common logic shared between steps.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the `_common_step` method.")

    def _get_gnn_layer(self, name: str, in_dim: int, out_dim: int) -> torch.nn.Module:
        """Factory function to create a GNN layer based on its name."""
        name = name.lower()
        if name == "gcn":
            return GCNConv(in_dim, out_dim)
        elif name == "cheb":
            k = self.hparams.gnn_k_hops
            return ChebConv(in_dim, out_dim, K=k)
        elif name == "tag":
            k = self.hparams.gnn_k_hops
            return TAGConv(in_dim, out_dim, K=k)
        elif name == "sgc":
            k = self.hparams.gnn_k_hops
            return SGConv(in_dim, out_dim, K=k)
        elif name == "ssg":
            k = self.hparams.gnn_k_hops
            alpha = self.hparams.gnn_alpha
            return SSGConv(in_dim, out_dim, alpha=alpha, K=k)
        else:
            raise ValueError(f"Unsupported GNN operator: {name}")


class GraphGNN_SPN_Model(BaseGNN_SPN_Model):
    """
    A GNN model for **graph-level** prediction tasks on SPNs.

    This model applies a series of specified GNN convolutions followed by a
    global pooling operation to produce a single embedding for the entire
    graph, which is then passed to an MLP for the final prediction.
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
        """
        Args:
            node_features_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden GCN layers.
            out_channels (int): Dimensionality of the final output (typically 1).
            num_layers (int): The number of GNN layers.
            gnn_operator_name (str): Name of the GNN operator to use.
            num_layers_mlp (int): Number of hidden layers in the final MLP.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            gnn_k_hops (int): The number of hops or filter size (K) for
                              operators like ChebConv, TAGConv, SGCConv, SSGConv.
            gnn_alpha (float): The alpha parameter for SSGConv.
        """
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        # Create GNN layers
        self.convs = torch.nn.ModuleList()
        # First layer
        self.convs.append(self._get_gnn_layer(gnn_operator_name, node_features_dim, hidden_dim))
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(self._get_gnn_layer(gnn_operator_name, hidden_dim, hidden_dim))

        # Create final MLP
        mlp_layers = []
        if num_layers_mlp > 0:
            mlp_layers.extend([Linear(hidden_dim, hidden_dim), ReLU()])
            for _ in range(num_layers_mlp - 1):
                mlp_layers.extend([Linear(hidden_dim, hidden_dim), ReLU()])
        mlp_layers.append(Linear(hidden_dim, out_channels))
        self.output_mlp = Sequential(*mlp_layers)

    def forward(self, batch: Data) -> torch.Tensor:
        """Defines the graph-level forward pass."""
        x, edge_index, edge_attr, batch_map = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        # Global pooling to get a graph-level embedding
        embedding = global_add_pool(x, batch_map)
        prediction = self.output_mlp(embedding)

        return prediction.squeeze(-1) if self.hparams.out_channels == 1 else prediction

    def _common_step(self, batch: Data, batch_idx: int, prefix: str) -> torch.Tensor:
        """Shared logic for training, validation, and testing."""
        y_pred = self(batch)
        y_true = batch.y.float()

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

        loss = F.mse_loss(y_pred, y_true)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self._log_metrics(prefix, y_pred, y_true)

        return loss


class NodeGNN_SPN_Model(BaseGNN_SPN_Model):
    """
    A GNN model for **node-level** prediction tasks on SPNs.

    This model applies a series of specified GNN convolutions to produce an
    embedding for each node. It includes logic to handle predictions for a
    specific subset of nodes (e.g., places or transitions) by using a mask
    derived from the input features.
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
        """
        Args:
            node_features_dim (int): Dimensionality of input node features.
            hidden_dim (int): Dimensionality of hidden GCN layers.
            out_channels (int): Dimensionality of the final output (typically 1).
            num_layers (int): The number of GNN layers.
            gnn_operator_name (str): Name of the GNN operator to use.
            num_layers_mlp (int): Number of hidden layers in the final MLP.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay (L2 penalty) for the optimizer.
            gnn_k_hops (int): The number of hops or filter size (K) for
                              operators like ChebConv, TAGConv, SGCConv, SSGConv.
            gnn_alpha (float): The alpha parameter for SSGConv.
        """
        super().__init__(learning_rate, weight_decay)
        self.save_hyperparameters()

        # Create GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(self._get_gnn_layer(gnn_operator_name, node_features_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(self._get_gnn_layer(gnn_operator_name, hidden_dim, hidden_dim))

        # Create final MLP
        mlp_layers = []
        if num_layers_mlp > 0:
            mlp_layers.extend([Linear(hidden_dim, hidden_dim), ReLU()])
            for _ in range(num_layers_mlp - 1):
                mlp_layers.extend([Linear(hidden_dim, hidden_dim), ReLU()])
        mlp_layers.append(Linear(hidden_dim, out_channels))
        self.output_mlp = Sequential(*mlp_layers)

    def forward(self, batch: Data) -> torch.Tensor:
        """Defines the node-level forward pass."""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        edge_weight = edge_attr.squeeze(-1) if edge_attr is not None and edge_attr.dim() > 1 else edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        # No global pooling; output node-level embeddings
        prediction = self.output_mlp(x)

        return prediction.squeeze(-1) if self.hparams.out_channels == 1 else prediction

    def _common_step(self, batch: Data, batch_idx: int, prefix: str) -> torch.Tensor:
        """Shared logic for training, validation, and testing."""
        y_pred_raw = self(batch)
        y_true = batch.y.float()

        # For node-level tasks, we must filter predictions to match ground truth.
        # The ground truth `y` may only correspond to a subset of nodes (e.g., places).
        place_mask = batch.x[:, 0].bool()
        transition_mask = batch.x[:, 1].bool()

        if place_mask.sum() == y_true.size(0):
            y_pred = y_pred_raw[place_mask]
        elif transition_mask.sum() == y_true.size(0):
            y_pred = y_pred_raw[transition_mask]
        else:
            raise ValueError(
                "Node-level prediction shape mismatch: The number of ground "
                f"truth labels ({y_true.size(0)}) does not match the count of "
                f"places ({place_mask.sum()}) or transitions ({transition_mask.sum()})."
            )

        if y_pred.shape != y_true.shape:
            raise ValueError(f"Final shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")

        loss = F.mse_loss(y_pred, y_true)
        self.log(
            f"{prefix}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        self._log_metrics(prefix, y_pred, y_true)

        return loss
