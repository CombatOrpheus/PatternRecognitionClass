from torch_geometric.nn.models import MLP, GCN


class MyMLP():
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        dropout: float,
        act: str = 'relu',
        norm: str = 'batch_norm',
        bias: bool = True,
    ):
        return MLP(
            in_channels=in_features,
            out_channels=1,
            hidden_channels=hidden_features,
            num_layers=num_layers,
            act=act,
            norm=norm,
            bias=bias)


class MyGCN():
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        dropout: float,
        act: str = 'relu',
        bias: bool = True,
    ):
        return GCN(
            in_channels=in_features,
            out_channels=1,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            bias=bias,
            normalize=False)


class CNN():
    pass
