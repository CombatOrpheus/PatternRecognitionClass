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
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, MeanAbsolutePercentageError


class LightningSPNModule(pl.LightningModule):
    """A base PyTorch Lightning module for SPN models.

    Encapsulates shared logic for training, validation, testing, optimizer
    configuration, and metric calculation for heterogeneous graph models.
    It tracks metrics both per node type and in aggregate.
    """

    def __init__(self, learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        """Initializes the LightningSPNModule.

        Args:
            learning_rate: The learning rate for the optimizer.
            weight_decay: The weight decay for the optimizer.
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.node_types = ["place", "transition"]
        for split in ["train", "val", "test"]:
            # Per-node-type metrics for detailed analysis
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

            # Aggregate metrics for overall performance and compatibility with analysis scripts
            agg_metrics = MetricCollection(
                {
                    "mae": MeanAbsoluteError(),
                    "mse": MeanSquaredError(),
                    "rmse": MeanSquaredError(squared=False),
                    "r2": R2Score(),
                    "mape": MeanAbsolutePercentageError(),
                }
            )
            setattr(self, f"{split}_agg_metrics", agg_metrics)

    def _common_step(self, batch: HeteroData, prefix: str) -> torch.Tensor:
        """Performs a common step for training, validation, and testing.

        This method computes the loss and logs metrics for each node type and
        in aggregate.

        Args:
            batch: The heterogeneous data batch.
            prefix: The prefix for logging (e.g., "train", "val", "test").

        Returns:
            The total loss for the batch.
        """
        output_dict = self(batch)
        total_loss = 0.0
        metrics_dict = getattr(self, f"{prefix}_metrics")
        agg_metrics_dict = getattr(self, f"{prefix}_agg_metrics")

        all_preds, all_trues = [], []

        for node_type, y_pred in output_dict.items():
            y_true = batch[node_type].y if hasattr(batch[node_type], "y") else None
            if y_true is None:
                continue

            loss = F.mse_loss(y_pred.squeeze(), y_true.float())
            total_loss += loss

            # Log per-node-type metrics
            metrics = metrics_dict[node_type]
            metrics.update(y_pred.squeeze(), y_true)
            self.log_dict({f"{prefix}/{node_type}/{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True)

            # Collect for aggregate metrics
            all_preds.append(y_pred.squeeze())
            all_trues.append(y_true)

        # **MODIFIED**: Compute and log aggregate metrics if applicable
        if all_preds:
            agg_preds = torch.cat(all_preds)
            agg_trues = torch.cat(all_trues)
            agg_metrics_dict.update(agg_preds, agg_trues)
            self.log_dict({f"{prefix}/{k}": v for k, v in agg_metrics_dict.items()}, on_step=False, on_epoch=True)

        if total_loss == 0.0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log(f"{prefix}/loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        return total_loss

    def training_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """The training step for the model.

        Args:
            batch: The training data batch.
            batch_idx: The index of the batch.

        Returns:
            The loss for the training batch.
        """
        return self._common_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """The validation step for the model.

        Args:
            batch: The validation data batch.
            batch_idx: The index of the batch.

        Returns:
            The loss for the validation batch.
        """
        return self._common_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> torch.Tensor:
        """The test step for the model.

        Args:
            batch: The test data batch.
            batch_idx: The index of the batch.

        Returns:
            The loss for the test batch.
        """
        loss = self._common_step(batch, "test")
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configures the optimizer for the model.

        Returns:
            A dictionary containing the optimizer.
        """
        return {
            "optimizer": torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        }

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """The forward pass of the model.

        This method must be implemented by subclasses.

        Args:
            batch: The input data batch.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class BaseHeteroGNN(LightningSPNModule, ABC):
    """An abstract base class for heterogeneous GNN models.

    It implements the shared forward pass logic and initializes the model's
    convolutional and linear layers.
    """

    def __init__(self, **kwargs):
        """Initializes the BaseHeteroGNN.

        Args:
            **kwargs: Keyword arguments for the model, including learning_rate
                and weight_decay.
        """
        super().__init__(kwargs.get("learning_rate", 1e-3), kwargs.get("weight_decay", 1e-5))
        self.save_hyperparameters()
        self.convs = ModuleList()
        self.lin = ModuleDict()

    @abstractmethod
    def _create_conv_layers(self) -> None:
        """Abstract method to create the GNN convolution layers.

        This method must be implemented by subclasses to build the specific
        GNN layer stack.
        """
        pass

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        """The forward pass for the heterogeneous GNN model.

        It applies the convolutional layers sequentially and then the final
        linear layers to produce node-level predictions.

        Args:
            batch: The heterogeneous data batch.

        Returns:
            A dictionary of tensors, where keys are node types and values are
            the predictions for the nodes of that type.
        """
        x_dict = batch.x_dict
        for conv in self.convs:
            extra_args = {}
            if "node_type_dict" in batch:
                extra_args["node_type_dict"] = batch.node_type_dict

            if "node_type_dict" in batch and "edge_type_dict" in batch:
                extra_args["node_type"] = batch.node_type_dict
                extra_args["edge_type"] = batch.edge_type_dict
            x_dict = conv(x_dict, batch.edge_index_dict, edge_attr_dict=batch.edge_attr_dict, **extra_args)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

        return {nt: self.lin[nt](x) for nt, x in x_dict.items()}


class RGAT_SPN_Model(BaseHeteroGNN):
    """An SPN evaluation model using Relational Graph Attention (RGAT) layers.

    This model is designed for heterogeneous graphs and uses RGATConv to handle
    different relation types.
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        num_layers: int,
        edge_dim: int,
        num_relations: int,
        **kwargs,
    ):
        """Initializes the RGAT_SPN_Model.

        Args:
            in_channels_dict: A dictionary mapping node types to their input
                feature dimensions.
            hidden_channels: The number of hidden channels in the GNN layers.
            out_channels: The number of output channels (prediction dimension).
            num_heads: The number of attention heads in the RGAT layers.
            num_layers: The number of GNN layers.
            edge_dim: The dimension of edge features.
            num_relations: The number of relation types.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self._create_conv_layers()

    def _create_conv_layers(self) -> None:
        """Creates the stack of RGAT convolutional layers."""
        in_channels = self.hparams.in_channels_dict
        layer_output_dim = self.hparams.hidden_channels * self.hparams.num_heads

        for i in range(self.hparams.num_layers):
            conv = HeteroConv(
                {
                    rel: RGATConv(
                        in_channels=(in_channels[rel[0]], in_channels[rel[2]]),
                        out_channels=self.hparams.hidden_channels,
                        heads=self.hparams.num_heads,
                        edge_dim=self.hparams.edge_dim,
                        num_relations=self.hparams.num_relations,
                    )
                    for rel in [("place", "to", "transition"), ("transition", "to", "place")]
                },
                aggr="sum",
            )
            self.convs.append(conv)
            in_channels = {key: layer_output_dim for key in in_channels.keys()}

        self.lin = ModuleDict({nt: Linear(layer_output_dim, self.hparams.out_channels) for nt in self.node_types})
