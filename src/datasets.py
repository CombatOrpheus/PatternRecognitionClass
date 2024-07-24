from itertools import starmap
from pathlib import Path
from typing import List, Iterator

import numpy as np
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_parser import get_reachability_graphs


def __pad_data__(iterator: Iterator) -> Iterator:
    graph_info = []
    labels = []
    for data in iterator:
        graph_info.append(data[0])
        labels.append(data[1])

    sizes = [net[0].shape for net in graph_info]
    max_size = max((size[1] for size in sizes))
    size_to_pad = ((size[0], max_size-size[1]) for size in sizes)
    padding = (np.zeros(size) for size in size_to_pad)
    graphs = (info[0] for info in graph_info)
    pair = zip(graphs, padding)
    padded_nets = [np.hstack((net, padding)) for net, padding in pair]
    new_info = ([padded_nets, *info[1:]] for info in graph_info)
    print(next(new_info))
    return zip(new_info, labels)


def __reduce_node_features__(
        graph_info: List[np.array],
        label: float
        ) -> (List[np.array], float):
    features = np.sum(graph_info[0], axis=1)
    return ([features, *graph_info[1:]], label)


def __to_data_no_node_features__(
        graph_info: List[np.array],
        label: float
        ) -> Data:
    return Data(
        edge_index=tensor(graph_info[1]).long(),
        edge_attr=graph_info[2][graph_info[3]],
        y=tensor(label))


def __to_data__(
        graph_info: List[np.array],
        label: float
        ) -> Data:
    return Data(
        x=tensor(graph_info[0]),
        edge_index=tensor(graph_info[1]),
        edge_attr=tensor(graph_info[3]),
        y=tensor(label))


def get_reachability_dataset(
        source: Path,
        batch_size=32,
        features: bool = True,
        pad_data: bool = True
        ) -> DataLoader:
    iterator = get_reachability_graphs(source)
    f = __to_data__ if features else __to_data_no_node_features__

    if pad_data and features:
        iterator = __pad_data__(iterator)
    if not pad_data and features:
        iterator = starmap(__reduce_node_features__, iterator)

    data = list(starmap(f, iterator))
    return DataLoader(data, batch_size)
