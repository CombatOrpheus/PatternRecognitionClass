from pathlib import Path

import numpy as np
from torch import from_numpy
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from src.BaseDataset import BaseDataset
from src.petri_nets import SPNData


def _reduce_features(net: SPNData) -> SPNData:
    net.spn = np.sum(net.spn, 0)
    return net


def _pad(net: SPNData, size: int) -> SPNData:
    padding = size - net.spn.shape[1]
    net.spn = np.pad(net.spn, (0, padding), 'constant')
    return net


class ReachabilityGraphDataset(BaseDataset):
    reduce: bool

    def __init__(self, path: Path, batch_size: int, reduce: bool = True):
        super().__init__(path, batch_size)
        self.reduce = reduce

    def create_dataloader(self, data):
        data = list(self._get_data())
        if self.reduce:
            nets = map(_reduce_features, data)
        else:
            size = max((net.spn.shape[1] for net in data))
            nets = (_pad(net, size) for net in data)

        data = [
            Data(
                x=from_numpy(net.spn),
                edge_index=from_numpy(net.reachability_graph_edges),
                edge_attr=from_numpy(net.average_firing_rates),
                y=net.average_tokens_network
            )
            for net in nets]
        self.data = data
        self.size = len(data)
        return DataLoader(data, self.batch_size, shuffle=True, drop_last=True)
