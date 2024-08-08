from itertools import starmap
from pathlib import Path
from typing import List, Iterable

import numpy as np
from torch import as_tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_parser import get_reachability_graphs


def __pad_node_features__(data: Iterable):
    info = []
    labels = []
    for data, label in data:
        info.append(data)
        labels.append(label)
    shapes = [np.shape(graph[0]) for graph in info]
    max_size = max(x[1] for x in shapes)
    sizes_to_pad = ((x[0], max_size - x[1]) for x in shapes)
    padding = map(np.zeros, sizes_to_pad)
    graphs = (x[0] for x in info)
    padded_data = map(np.hstack, zip(graphs, padding))
    new_info = [[padded_data, *graph[1:]] for graph in info]
    return max_size, zip(new_info, labels)


def __to_data__(
    graph_info: List[np.array],
    label: float
) -> Data:
    return Data(
        x=as_tensor(graph_info[0]).float(),
        edge_index=as_tensor(graph_info[1]).view(2, -1).long(),
        edge_attr=as_tensor(graph_info[3][graph_info[2]]).float(),
        y=as_tensor(label),
        num_nodes=graph_info[0].shape[0])


def __to_data_reduced_features__(
        graph_info: List[np.array],
        label: float
) -> Data:
    return Data(
        x=as_tensor(np.sum(graph_info[0], axis=1)).view(-1, 1).float(),
        edge_index=as_tensor(graph_info[1]).view(2, -1).long(),
        edge_attr=as_tensor(graph_info[3][graph_info[2]]).float(),
        y=as_tensor(label),
        num_nodes=graph_info[0].shape[0])


def get_reachability_dataset(
    source: Path,
    reduce_node_features: bool = True,
    batch_size=16
) -> DataLoader:
    data = get_reachability_graphs(source)
    f = __to_data_reduced_features__ if reduce_node_features else __to_data__
    if not reduce_node_features:
        size, data = __pad_node_features__(data)

    data_iterator = list(starmap(f, data))
    loader = DataLoader(data_iterator, batch_size=batch_size, shuffle=True)
    loader.num_features = 1 if reduce_node_features else size

    return loader
