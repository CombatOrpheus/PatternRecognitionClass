import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GCN, MLP, ChebConv
from torch_geometric.utils import scatter


class MLPReadout(nn.Module):
    """Layer for reading out the graph representation into the final regression
    value
    """

    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        layer_sizes = [input_dim//2**layer for layer in range(L)]
        layer_sizes.append(output_dim)
        self.layers = MLP(layer_sizes, bias=True)

    def forward(self, x):
        return self.layers(x)


class Petri_GCN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_features: int,
                 num_layers: int,
                 dropout: float = .0,
                 act: str = 'relu',
                 norm: str = None,
                 readout_layers: int = 2):
        super().__init__()
        self.GNN = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_features,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            norm=norm)
        self.MLP_layer = MLPReadout(hidden_features, 1)
        self.mae_loss = F.l1_loss
        self.mre_loss = F.mse_loss

    def forward(self, g):
        x = self.GNN(g.x, g.edge_index)
        x = self.MLP_layer(x)
        return scatter(x, g.batch, dim=0, reduce='mean')

    def loss(self, scores, targets):
        return self.mae_loss(scores, targets)


class Petri_Cheb_GNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_features: int,
                 num_layers: int,
                 filter_size: int = 3,
                 norm: str = 'sym',
                 readout_layers: int = 2):
        super().__init__()
        layers = [ChebConv(in_channels, hidden_features, filter_size, norm)]
        layers.extend([
            ChebConv(hidden_features, hidden_features, filter_size, norm)
            for _ in range(num_layers)])

        self.layers = nn.ModuleList(layers)
        self.readout = MLPReadout(hidden_features, 1, readout_layers)
        self.mae_loss = F.l1_loss
        self.mre_loss = F.mse_loss

    def forward(self, data: Batch):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_attr

        y = self.layers[0](x, edge_index, edge_weight)
        for layer in self.layers[1:]:
            y = layer(y, edge_index, edge_weight)
        y = self.readout(y)
        return scatter(y, data.batch, dim=0, reduce='mean')

    def loss(self, scores, targets):
        return self.mae_loss(scores, targets)
