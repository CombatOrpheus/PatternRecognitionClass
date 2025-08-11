from pathlib import Path
from typing import List
from itertools import starmap

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import as_tensor, from_numpy
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_parser import get_average_tokens, get_steady_state


def __pad_features__(info: List[np.array], pad_to: int) -> Tensor:
    features = info[0]
    size = np.shape(features)[1]
    tensor = from_numpy(features).float()
    return [F.pad(tensor, (pad_to - size, 0)), *info[1:]]


def __reduce_features__(info: List[np.array]):
    features = from_numpy(info[0]).float()
    return [torch.sum(features, 1, keepdim=True), *info[1:]]


def __to_data__(
    info: List[np.array],
    label,
) -> Data:
    return Data(
        x=info[0],
        edge_index=as_tensor(info[1]).view(2, -1).long(),
        edge_attr=as_tensor(info[3][info[2]]).float(),
        y=as_tensor(label),
        num_nodes=info[0].shape[0],
    )


def __steady_state_data__(info: List) -> Data:
    return Data(
        x=info[0],
        edge_index=from_numpy(info[1]).view(2, -1).long(),
        edge_attr=as_tensor(info[3][info[2]]).float(),
        y=from_numpy(info[5]).float(),
        num_nodes=info[0].shape[0],
    )


def get_average_tokens_dataset(
    source: Path, reduce_features: bool = True, batch_size=16, network: bool = True
) -> DataLoader:
    data = get_average_tokens(source, network)
    size = 1
    if reduce_features:
        iterator = ((__reduce_features__(graph), label) for graph, label in data)
    else:
        data = list(data)
        size = max(info[0].shape[1] for info, _ in data)
        iterator = ((__pad_features__(graph, size), label) for graph, label in data)

    loader = DataLoader(list(starmap(__to_data__, iterator)), batch_size=batch_size, shuffle=True)
    loader.num_features = size
    return loader


def get_steady_state_dataset(source: Path, batch_size: int = 16):
    data = list(get_steady_state(source))
    iterator = ((__steady_state_data__(graph), label) for graph, label in data)
    loader = DataLoader(list(iterator), batch_size=batch_size, shuffle=True)
    loader.num_features = 1
    return loader
