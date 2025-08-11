import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, mean
from torch_geometric.data import Batch
from torch_geometric.nn import GCN, MLP, GraphConv
from torch_geometric.utils import scatter


LAYER_DICT = {"GCN": GCN, "GraphConv": GraphConv}
LAYERS = list(LAYER_DICT.keys())


def __relative_error__(input: Tensor, target: Tensor):
    return mean(F.l1_loss(input, target, reduction="none") / target) * 100


class MLPReadout(nn.Module):
    """Layer for reading out the graph representation into the final regression
    value
    """

    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        layer_sizes = [input_dim // 2**layer for layer in range(L)]
        layer_sizes.append(output_dim)
        self.Model = MLP(layer_sizes, bias=True)

    def forward(self, x):
        return self.Model(x)


class Petri_GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        readout_layers: int = 2,
        mae: bool = True,
        edge_attr: bool = True,
    ):
        super().__init__()
        self.GNN = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        self.Readout_Layer = MLPReadout(hidden_channels, 1)
        self.loss_function = F.l1_loss if mae else __relative_error__
        self.edge_attr = edge_attr

    def forward(self, g):
        x = self.GNN(g.x, g.edge_index, g.edge_attr)
        x = self.Readout_Layer(x)
        return scatter(x, g.batch, dim=0, reduce="mean")

    def loss(self, scores, targets):
        return self.loss_function(scores, targets)


class Petri_GraphConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        readout_layers: int = 2,
        mae: bool = True,
    ):
        super().__init__()
        layers = [GraphConv(in_channels, hidden_channels)]
        for _ in range(num_layers):
            layers.append(GraphConv(hidden_channels, hidden_channels))

        self.layers = nn.ModuleList(layers)
        self.Readout_Layer = MLPReadout(hidden_channels, 1, readout_layers)
        self.loss_function = F.l1_loss if mae else __relative_error__

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        edge_attr = g.edge_attr

        y = self.layers[0](x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            y = layer(y, edge_index, edge_attr)
        y = self.Readout_Layer(y)
        return scatter(y, g.batch, dim=0, reduce="mean")

    def loss(self, scores, targets):
        return self.loss_function(scores, targets)


class Petri_PerPlace_Average(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, mae: bool = True):
        super().__init__()
        self.GNN = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=1,
        )
        self.loss_function = F.l1_loss if mae else __relative_error__

    def forward(self, g):
        x = g.x
        edge_index = g.edge_index
        edge_attr = g.edge_attr

        x = self.GNN(x, edge_index, edge_attr)
        return scatter(x, g.batch, dim=0, reduce="mean")

    def loss(self, scores, targets):
        return self.loss_function(scores, targets)
