"""This module defines the data structures and loading utilities for
Stochastic Petri Net (SPN) data.

It includes:
- `SPNData`: A dataclass to hold all information related to a single SPN,
  including its structure, reachability graph, and analysis results.
- Conversion methods to represent the SPN as a homogeneous or heterogeneous
  graph suitable for GNNs.
- Helper functions to load SPN data from JSON-L files, either eagerly or
  lazily.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, Union, Iterator

import numpy as np
import torch
from torch_geometric.data import HeteroData

# Define the Literal for the specific analysis result labels
SPNAnalysisResultLabel = Literal[
    "average_firing_rates",
    "steady_state_probabilities",
    "token_probability_density_function",
    "average_tokens_per_place",
    "average_tokens_network",
]


@dataclass
class SPNData:
    """Represents a Stochastic Petri Net and its associated data.

    This class encapsulates the structure of an SPN (input, output, and initial
    marking), its reachability graph, and various performance metrics derived
    from analysis. It also provides methods to convert the SPN into graph
    representations for use with GNNs.

    Attributes:
        spn: The SPN in a compact matrix form [I; O; M_0].
        incidence_matrix: The incidence matrix C = O - I.
        reachability_graph_nodes: List of markings in the reachability graph.
        reachability_graph_edges: Edges of the reachability graph.
        transition_indices: Indices of transitions fired to generate markings.
        average_firing_rates: Average firing rates (lambda) for each transition.
        steady_state_probabilities: Steady-state probabilities for each marking.
        token_probability_density_function: PDF of tokens per place.
        average_tokens_per_place: Average number of tokens for each place.
        average_tokens_network: Average number of tokens in the whole network.
        num_node_features: The number of features for nodes in the homogeneous graph.
    """

    spn: np.ndarray
    incidence_matrix: np.ndarray
    reachability_graph_nodes: List[np.ndarray]
    reachability_graph_edges: np.ndarray
    transition_indices: np.ndarray
    average_firing_rates: np.ndarray
    steady_state_probabilities: np.ndarray
    token_probability_density_function: np.ndarray
    average_tokens_per_place: np.ndarray
    average_tokens_network: float
    num_node_features: int

    def __init__(self, data: dict):
        """Initializes and validates the SPNData object from a dictionary.

        Args:
            data: A dictionary containing all the raw SPN data loaded from a
                  JSON-L file.

        Raises:
            ValueError: If required keys are missing from the input dictionary.
        """
        # Ensure data exists and has expected keys before accessing
        required_keys = [
            "petri_net",
            "arr_vlist",
            "arr_edge",
            "arr_tranidx",
            "spn_labda",
            "spn_steadypro",
            "spn_markdens",
            "spn_allmus",
            "spn_mu",
        ]
        if not all(key in data for key in required_keys):
            missing = [key for key in required_keys if key not in data]
            raise ValueError(f"Missing required keys in input data: {missing}")

        self.spn = np.array(data["petri_net"])
        # Basic validation for spn shape
        if self.spn.ndim != 2 or self.spn.shape[1] % 2 != 1:
            print(f"Warning: Unexpected SPN shape {self.spn.shape}. Expected (places, 2*transitions + 1).")

        self.incidence_matrix = to_incidence_matrix(self.spn)
        self.reachability_graph_nodes = [np.array(marking) for marking in data["arr_vlist"]]
        self.reachability_graph_edges = np.array(data["arr_edge"])
        self.transition_indices = np.array(data["arr_tranidx"])
        self.average_firing_rates = np.array(data["spn_labda"]).squeeze()
        self.steady_state_probabilities = np.array(data["spn_steadypro"]).squeeze()
        self.token_probability_density_function = np.array(data["spn_markdens"])
        self.average_tokens_per_place = np.array(data["spn_allmus"]).squeeze()
        self.average_tokens_network = data["spn_mu"]

        # Basic shape validation for analysis results
        num_transitions = self.spn.shape[1] // 2
        num_places = self.spn.shape[0]
        if self.average_firing_rates.shape[0] != num_transitions:
            print(
                f"Warning: Mismatch in average_firing_rates shape {self.average_firing_rates.shape} and number of transitions {num_transitions}."
            )
        if self.average_tokens_per_place.shape[0] != num_places:
            print(
                f"Warning: Mismatch in average_tokens_per_place shape {self.average_tokens_per_place.shape} and number of places {num_places}."
            )

        self.num_node_features = 4

    def to_information(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts the Petri net into a homogeneous graph representation.

        This method creates a graph where both places and transitions are nodes.
        It's suitable for use with homogeneous GNN models.

        Node Features (4 per node):
        - Is_Place (one-hot)
        - Is_Transition (one-hot)
        - Initial Marking (for places)
        - Firing Rate (for transitions)

        Returns:
            A tuple containing node features, edge features (arc weights), and
            edge pairs (source, target indices).
        """
        num_places = self.spn.shape[0]
        # Assuming spn has shape (num_places, 2 * num_transitions + 1)
        num_transitions = self.spn.shape[1] // 2
        num_nodes = num_places + num_transitions

        # Extract components from the SPN matrix
        pre_conditions = self.spn[:, :num_transitions]  # Input matrix I (P -> T)
        post_conditions = self.spn[:, num_transitions : 2 * num_transitions]  # Output matrix O (T -> P)
        initial_marking = self.spn[:, -1]  # M_0
        lambdas = self.average_firing_rates  # Transition rates

        # --- Node Features ---
        # Initialize node features with zeros. Dimension is now 4.
        node_features = np.zeros((num_nodes, 4), dtype=float)

        # Assign one-hot encoding for node type
        # First column: Is_Place (1 for places, 0 for transitions)
        node_features[0:num_places, 0] = 1.0
        # Second column: Is_Transition (1 for transitions, 0 for places)
        node_features[num_places:, 1] = 1.0

        # Assign type-specific features
        # Third column: Initial Marking (only for places, 0 for transitions)
        node_features[0:num_places, 2] = initial_marking
        # Fourth column: Firing Rate (only for transitions, 0 for places)
        node_features[num_places:, 3] = lambdas

        # --- Edges and Edge Features (Homogeneous Graph Structure) ---
        # Node indices for places and transitions in the homogeneous graph
        place_node_indices = np.arange(num_places)
        transition_node_indices = np.arange(num_places, num_nodes)

        # Edges from Places to Transitions (P -> T)
        # Find where there are arcs from places to transitions
        p_in_idx, t_in_idx = np.nonzero(pre_conditions)
        # Source nodes are places, target nodes are transitions
        src_in = place_node_indices[p_in_idx]
        tgt_in = transition_node_indices[t_in_idx]
        edges_in = np.stack([src_in, tgt_in], axis=1)
        # Edge features are the arc weights
        weights_in = pre_conditions[p_in_idx, t_in_idx]

        # Edges from Transitions to Places (T -> P)
        # Find where there are arcs from transitions to places
        p_out_idx, t_out_idx = np.nonzero(post_conditions)
        # Source nodes are transitions, target nodes are places
        src_out = transition_node_indices[t_out_idx]
        tgt_out = place_node_indices[p_out_idx]
        edges_out = np.stack([src_out, tgt_out], axis=1)
        # Edge features are the arc weights
        weights_out = post_conditions[p_out_idx, t_out_idx]

        # Combine P->T and T->P edges and features
        if edges_in.size > 0 and edges_out.size > 0:
            edge_pairs = np.vstack((edges_in, edges_out)).astype(int).transpose()  # Ensure int dtype
            edge_features = np.concatenate((weights_in, weights_out)).reshape(-1, 1).astype(float)  # Ensure float dtype
        elif edges_in.size > 0:
            edge_pairs = edges_in.astype(int).transpose()
            edge_features = weights_in.reshape(-1, 1).astype(float)
        elif edges_out.size > 0:
            edge_pairs = edges_out.astype(int).transpose()
            edge_features = weights_out.reshape(-1, 1).astype(float)
        else:
            # Handle case with no edges
            edge_pairs = np.empty((2, 0), dtype=int)
            edge_features = np.empty((0, 1), dtype=float)

        # Return node features, edge features, and edge pairs
        return node_features, edge_features, edge_pairs

    def to_hetero_information(self) -> "HeteroData":
        """Converts the Petri net into a PyG HeteroData object.

        This representation is suitable for heterogeneous GNN models, as it
        distinguishes between 'place' and 'transition' nodes and the directed
        edges between them. It also includes integer-based type mappings
        required by models like HEATConv.

        Returns:
            A PyG HeteroData object representing the SPN.
        """
        data = HeteroData()

        num_places = self.spn.shape[0]
        num_transitions = self.spn.shape[1] // 2

        # Node Features
        data['place'].x = torch.from_numpy(self.spn[:, -1]).float().view(-1, 1)
        data['transition'].x = torch.from_numpy(self.average_firing_rates).float().view(-1, 1)

        # Edges and Edge Features
        pre_conditions = self.spn[:, :num_transitions]
        post_conditions = self.spn[:, num_transitions: 2 * num_transitions]

        p_in_idx, t_in_idx = np.nonzero(pre_conditions)
        pt_edge_index = np.stack([p_in_idx, t_in_idx], axis=0)
        pt_edge_attr = pre_conditions[p_in_idx, t_in_idx]
        data['place', 'to', 'transition'].edge_index = torch.from_numpy(pt_edge_index).long()
        data['place', 'to', 'transition'].edge_attr = torch.from_numpy(pt_edge_attr).float().view(-1, 1)

        p_out_idx, t_out_idx = np.nonzero(post_conditions)
        tp_edge_index = np.stack([t_out_idx, p_out_idx], axis=0)
        tp_edge_attr = post_conditions[p_out_idx, t_out_idx]
        data['transition', 'to', 'place'].edge_index = torch.from_numpy(tp_edge_index).long()
        data['transition', 'to', 'place'].edge_attr = torch.from_numpy(tp_edge_attr).float().view(-1, 1)

        # **BUG FIX**: Add integer-based type mappings required by HEATConv.
        # Node type mapping: 'place' -> 0, 'transition' -> 1
        data['place'].node_type = torch.zeros(num_places, dtype=torch.long)
        data['transition'].node_type = torch.ones(num_transitions, dtype=torch.long)

        # Edge type mapping: ('place', 'to', 'transition') -> 0, ('transition', 'to', 'place') -> 1
        num_pt_edges = pt_edge_index.shape[1]
        num_tp_edges = tp_edge_index.shape[1]
        data['place', 'to', 'transition'].edge_type = torch.zeros(num_pt_edges, dtype=torch.long)
        data['transition', 'to', 'place'].edge_type = torch.ones(num_tp_edges, dtype=torch.long)

        return data

    def get_analysis_result(self, label: SPNAnalysisResultLabel) -> Union[np.ndarray, float]:
        """Fetches a specific analysis result by its label.

        Args:
            label: The name of the analysis result to fetch. Must be one of
                   the values defined in `SPNAnalysisResultLabel`.

        Returns:
            The value of the requested analysis result, which can be either
            a NumPy array or a float.

        Raises:
            AttributeError: If the label is not a valid attribute of the class.
        """
        # Use getattr to dynamically access the attribute based on the label string
        if not hasattr(self, label):
            raise AttributeError(f"Invalid analysis result label: {label}")
        return getattr(self, label)


def to_incidence_matrix(pn: np.ndarray) -> np.ndarray:
    """Converts a Petri net representation [I; O; M_0] to its incidence matrix C.

    The incidence matrix is calculated as C = O - I, where O is the output
    matrix and I is the input matrix.

    Args:
        pn: A NumPy array representing the Petri net in the format [I; O; M_0].
            It must be 2D and have an odd number of columns.
            Shape: (num_places, 2 * num_transitions + 1).

    Returns:
        The incidence matrix (C) as a NumPy array.
        Shape: (num_places, num_transitions).

    Raises:
        ValueError: If the input matrix is not 2D or has an even number of columns.
    """
    if pn.ndim != 2:
        raise ValueError("Input Petri net matrix must be 2-dimensional.")
    num_transitions = pn.shape[1] // 2
    if pn.shape[1] % 2 != 1:
        raise ValueError("Input Petri net matrix must have an odd number of columns (2*transitions + 1).")

    pre_conditions = pn[:, :num_transitions]
    post_conditions = pn[:, num_transitions : 2 * num_transitions]
    incidence_matrix = post_conditions - pre_conditions
    return incidence_matrix


def load_spn_data_from_files(file_paths: Union[Path, List[Path]]) -> List[SPNData]:
    """Loads a list of SPNData objects from one or more JSON-L files.

    This function reads all data into memory and returns a list. For very large
    datasets where memory is a concern, consider using `load_spn_data_lazily`.

    Args:
        file_paths: A single Path object or a list of Path objects pointing
                    to the JSON-L data files.

    Returns:
        A list of SPNData instances, one for each line in the files.
    """
    # This is a convenience wrapper around the lazy loader to materialize the list.
    return list(load_spn_data_lazily(file_paths))


def load_spn_data_lazily(file_paths: Union[Path, List[Path]]) -> Iterator[SPNData]:
    """Lazily loads SPNData objects from one or more JSON-L files.

    This function is memory-efficient as it yields one SPNData object at a time
    using a generator, avoiding loading the entire dataset into memory.

    Args:
        file_paths: A single Path or a list of Paths to the JSON-L data files.

    Yields:
        An iterator of SPNData instances.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
        json.JSONDecodeError: If a line in a file is not valid JSON.
    """
    # Normalize input to a list of paths for consistent processing
    if isinstance(file_paths, Path):
        paths = [file_paths]
    else:
        paths = file_paths

    for file_path in paths:
        if not file_path.is_file():
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        print(f"Streaming data from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Each line is a separate JSON object
                data_dict = json.loads(line)
                # 'yield' turns this function into a generator
                yield SPNData(data_dict)
