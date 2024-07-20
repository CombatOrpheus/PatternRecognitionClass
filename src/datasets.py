from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
from data_parser import get_reachability_graphs
from typing import List
import numpy as np


def __reachability_to_data__(reachability_graph: List[np.array], label: float) -> Data:
    return Data(
        x=reachability_graph[0],
        edge_index=reachability_graph[1],
        edge_attr=reachability_graph[3],
        y=label)


def get_reachability_dataset(source: Path, batch_size=32) -> DataLoader:
    iterator = get_reachability_graphs(source)
    data = list(map(__reachability_to_data__, iterator))
    return DataLoader(data, batch_size)
