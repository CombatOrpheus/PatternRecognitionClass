from itertools import starmap
from pathlib import Path
from typing import List, Iterable

import numpy as np
from torch import tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_parser import get_reachability_graphs


def __pad_node_features__(data: Iterable):
    pass


def __to_data_reduced_features__(
        graph_info: List[np.array],
        label: float
) -> Data:
    return Data(
        x=tensor(np.sum(graph_info[0], axis=1)).view(-1, 1).float(),
        edge_index=tensor(graph_info[1]).view(2, -1).long(),
        edge_attr=tensor(graph_info[3][graph_info[2]]).float(),
        y=tensor(label),
        num_nodes=graph_info[0].shape[0])


def get_reachability_dataset(
    source: Path,
    reduce_node_features: bool = True,
    batch_size=16
) -> DataLoader:
    data = get_reachability_graphs(source)
    if reduce_node_features:
        data_iterator = list(starmap(__to_data_reduced_features__, data))
    else:
        data_iterator = __pad_node_features__(data)

    return DataLoader(data_iterator, batch_size=batch_size)
