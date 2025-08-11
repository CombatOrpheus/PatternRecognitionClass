from pathlib import Path

import numpy as np
from torch import from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.BaseDataset import BaseDataset
from src.PetriNets import SPNData


def _reduce_features(net: SPNData) -> SPNData:
    net.spn = np.sum(net.spn, 0)
    return net


def _pad(net: SPNData, size: int) -> SPNData:
    padding = size - net.spn.shape[1]
    net.spn = np.pad(net.spn, (0, padding), "constant")
    return net


class ReachabilityGraphDataset(BaseDataset):
    reduce: bool

    def __init__(self, path: Path, batch_size: int, reduce: bool = True):
        super().__init__(path, batch_size)
        self.reduce = reduce
        self._create_dataloader()

    def _create_dataloader(self) -> None:
        data = list(self._get_data())
        size = 1
        if self.reduce:
            nets = map(_reduce_features, data)
        else:
            size = max((net.spn.shape[1] for net in data))
            nets = (_pad(net, size) for net in data)

        data = [
            Data(
                x=from_numpy(net.spn),
                edge_index=from_numpy(net.reachability_graph_nodes),
                edge_attr=from_numpy(net.transition_indices[net.reachability_graph_edges]),
                y=net.average_tokens_network,
                num_nodes=net.spn.shape[0],
            )
            for net in nets
        ]
        self.data = data
        self.size = len(data)
        self.features = size
        self.loader = DataLoader(data, self.batch_size, shuffle=True, drop_last=True)
