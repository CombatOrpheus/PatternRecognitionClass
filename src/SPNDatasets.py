"""
This module defines custom PyTorch Geometric Dataset classes for handling
the pre-processing and loading of SPN data.

**MODIFIED**: This version creates unique processed filenames based on the
raw input file to prevent data caching conflicts and adds properties to
easily access dataset metadata like the number of features.
"""

from pathlib import Path
from typing import List, Callable, Optional, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from src.PetriNets import load_spn_data_lazily, SPNAnalysisResultLabel


class BaseSPNDataset(InMemoryDataset):
    """
    An abstract base class for our custom SPN datasets.
    It handles the file finding and basic setup.
    """

    def __init__(
        self,
        root: Union[str, Path],
        raw_data_dir: Union[str, Path],
        raw_file_name: Union[str, Path],
        label_to_predict: SPNAnalysisResultLabel,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_file_name = Path(raw_file_name)
        self.label_to_predict = label_to_predict
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return str(self.raw_data_dir)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.raw_file_name.name]

    @property
    def processed_file_names(self) -> str:
        sanitized_name = self.raw_file_name.stem
        return f"data_{sanitized_name}_{self.label_to_predict}.pt"

    def download(self):
        pass

    def process(self):
        raise NotImplementedError("Subclasses must implement the process method.")

    @property
    def num_node_features(self) -> Union[int, Dict[str, int]]:
        """Returns the number of features per node type."""
        if not hasattr(self, "_data") or self._data is None:
            return 0
        if self.data.is_hetero:
            return {store._key: store.num_node_features for store in self.data.node_stores}
        return self.data.num_node_features

    @property
    def num_edge_features(self) -> Union[int, Dict[str, int]]:
        """Returns the number of features per edge type."""
        if not hasattr(self, "_data") or self._data is None:
            return 0
        if self.data.is_hetero:
            return {store._key: store.num_edge_features for store in self.data.edge_stores}
        return self.data.num_edge_features


class HomogeneousSPNDataset(BaseSPNDataset):
    """Creates a pre-processed dataset of homogeneous graph representations of SPNs."""

    def process(self):
        data_list = []
        raw_path = Path(self.raw_paths[0])
        for spn_data in tqdm(load_spn_data_lazily(Path(raw_path)), desc=f"Processing {self.raw_file_name}"):
            node_features, edge_features, edge_pairs = spn_data.to_information()
            label = spn_data.get_analysis_result(self.label_to_predict)
            data = Data(
                x=torch.from_numpy(node_features).float(),
                edge_attr=torch.from_numpy(edge_features).float(),
                edge_index=torch.from_numpy(edge_pairs).long(),
                y=torch.tensor([label] if not isinstance(label, np.ndarray) else label).float(),
            )
            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class HeterogeneousSPNDataset(BaseSPNDataset):
    """Creates a pre-processed dataset of heterogeneous graph representations of SPNs."""

    def process(self):
        data_list = []
        raw_path = Path(self.raw_paths[0])
        for spn_data in tqdm(load_spn_data_lazily(Path(raw_path)), desc=f"Processing {self.raw_file_name}"):
            data = spn_data.to_hetero_information()
            label = spn_data.get_analysis_result(self.label_to_predict)
            if self.label_to_predict == "average_tokens_per_place":
                data["place"].y = torch.tensor(label).float()
            else:
                # For graph-level labels
                data.y = torch.tensor([label]).float()
            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class ReachabilityGraphInMemoryDataset(BaseSPNDataset):
    """Creates a pre-processed dataset from SPN reachability graphs, with padding."""

    def process(self):
        data_list = []
        raw_path = Path(self.raw_paths[0])
        all_spn_data = list(load_spn_data_lazily(Path(raw_path)))
        max_num_places = max(spn.spn.shape[0] for spn in all_spn_data)

        for net in tqdm(all_spn_data, desc=f"Processing {self.raw_file_name}"):
            node_features = torch.tensor(np.array(net.reachability_graph_nodes), dtype=torch.float)
            padding_size = max_num_places - node_features.shape[1]
            if padding_size > 0:
                node_features = F.pad(node_features, (0, padding_size), "constant", 0)
            edge_index = torch.tensor(net.reachability_graph_edges, dtype=torch.long).t().contiguous()
            firing_rates = torch.tensor(net.average_firing_rates, dtype=torch.float)
            transition_indices = torch.tensor(net.transition_indices, dtype=torch.long)
            edge_attr = firing_rates[transition_indices].view(-1, 1)
            label = net.get_analysis_result(self.label_to_predict)
            data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]).float())
            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
