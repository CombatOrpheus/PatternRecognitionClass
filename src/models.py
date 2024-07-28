import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, MLP
from torch.utils import scatter


class MLPReadout(nn.Module):
    """Layer for reading out the graph representation into the final regression
    value
    """

    def __init__(self, input_dim, output_dim, L=2):
        super.__init__()
        layer_sizes = [input_dim//2**layer for layer in range(L)]
        layer_sizes.append(1)
        self.layers = MLP(layer_sizes, bias=True)

    def forward(self, x):
        return self.layers(x)


class Petri_GCN(nn.Module):
    def __init__(self,
                 in_features: int,
                 h_dim: int,
                 num_layers: int,
                 dropout: float,
                 act: str,
                 norm: str):
        self.GNN = GCN(
            in_features=in_features,
            hidden_channels=h_dim,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            norm=norm)
        self.MLP_layer = MLPReadout(self.h_dim, 1)
        self.mae_loss = nn.L1Loss()
        self.mre_loss = nn.MSELoss()

    def forward(self, g):
        x = self.GNN(g.x, g.edge_index)
        x = self.MLP_layer(x)
        return scatter(x, g.batch, dim=0, reduce='mean')

    def loss(self, scores, targets):
        loss = nn.L1Loss()(scores, targets)
        return loss
