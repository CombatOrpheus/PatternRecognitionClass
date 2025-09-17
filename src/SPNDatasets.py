"""This module defines custom PyTorch Geometric `Dataset` classes for handling
the preprocessing and loading of SPN data.

It provides a base class and specific implementations for homogeneous,
heterogeneous, and reachability graph representations. A key feature is the
creation of unique processed filenames based on the raw input file to prevent
data caching conflicts.
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
    """An abstract base class for custom SPN datasets.

    This class handles the file finding, basic setup, and provides properties
    for accessing dataset metadata. Subclasses must implement the `process`
    method.
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
        """Initializes the BaseSPNDataset.

        Args:
            root: The root directory where the dataset should be stored.
            raw_data_dir: The directory where the raw data is located.
            raw_file_name: The name of the raw data file.
            label_to_predict: The specific analysis result to use as the label.
            transform: A function/transform that takes in an object and
                returns a transformed version.
            pre_transform: A function/transform that takes in an object and
                returns a transformed version.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_file_name = Path(raw_file_name)
        self.label_to_predict = label_to_predict
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """The directory where the raw data is stored."""
        return str(self.raw_data_dir)

    @property
    def raw_file_names(self) -> List[str]:
        """The name of the raw file."""
        return [self.raw_file_name.name]

    @property
    def processed_file_names(self) -> str:
        """The name of the processed file, unique to the raw file and label."""
        sanitized_name = self.raw_file_name.stem
        return f"data_{sanitized_name}_{self.label_to_predict}.pt"

    def download(self):
        """This dataset does not support downloading."""
        pass

    def process(self):
        """Abstract method for processing the raw data."""
        raise NotImplementedError("Subclasses must implement the process method.")

    @property
    def num_node_features(self) -> Union[int, Dict[str, int]]:
        """The number of features per node type."""
        if not hasattr(self, "_data") or self._data is None:
            return 0
        if self.data.is_hetero:
            return {store._key: store.num_node_features for store in self.data.node_stores}
        return self.data.num_node_features

    @property
    def num_edge_features(self) -> Union[int, Dict[str, int]]:
        """The number of features per edge type."""
        if not hasattr(self, "_data") or self._data is None:
            return 0
        if self.data.is_hetero:
            return {store._key: store.num_edge_features for store in self.data.edge_stores}
        return self.data.num_edge_features


class HomogeneousSPNDataset(BaseSPNDataset):
    """Creates a preprocessed dataset of homogeneous graph representations of SPNs."""

    def process(self):
        """Processes the raw JSON-L data into a list of homogeneous `Data` objects."""
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
    """Creates a preprocessed dataset of heterogeneous graph representations of SPNs."""

    def process(self):
        """Processes the raw JSON-L data into a list of `HeteroData` objects."""
        data_list = []
        raw_path = Path(self.raw_paths[0])
        for spn_data in tqdm(load_spn_data_lazily(Path(raw_path)), desc=f"Processing {self.raw_file_name}"):
            data = spn_data.to_hetero_information()
            label = spn_data.get_analysis_result(self.label_to_predict)
            if self.label_to_predict == "average_tokens_per_place":
                data["place"].y = torch.tensor(label).float()
                if hasattr(data, "y"):
                    del data.y
            else:
                # For graph-level labels
                data.y = torch.tensor([label] if not isinstance(label, np.ndarray) else label).float()
            data_list.append(data)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])


class ReachabilityGraphInMemoryDataset(BaseSPNDataset):
    """Creates a preprocessed dataset from SPN reachability graphs.

    This class processes the reachability graph of each SPN, pads the node
    features to ensure consistency across the dataset, and stores the result
    as a list of `Data` objects.
    """

    def process(self):
        """Processes raw data to create reachability graph `Data` objects."""
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
