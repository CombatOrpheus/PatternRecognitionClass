import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Literal, Union, Iterator

import numpy as np

# Define the Literal for the specific analysis result labels
SPNAnalysisResultLabel = Literal[
    "average_firing_rates",
    "steady_state_probabilities",
    "token_probability_density_function",
    "average_tokens_per_place",
    "average_tokens_network"
]


@dataclass
class SPNData:
    """
    Represents a Stochastic Petri net with its incidence matrix, output matrix,
    initial marking, and additional analysis results.

    Attributes:
        spn: The Stochastic Petri Net in compound form (column-based) [I; O; M_0].
             Shape: (num_places, 2 * num_transitions + 1).
        incidence_matrix: The incidence matrix (C = O - I).
                          Shape: (num_places, num_transitions).
        reachability_graph_nodes: List of markings in the reachability graph.
        reachability_graph_edges: Edges of the reachability graph.
        transition_indices: Transition indices that fired to generate each
                            marking in `reachability_graph_nodes`.
        average_firing_rates: Average firing rates (lambda) for each transition.
                              Shape: (num_transitions,).
        steady_state_probabilities: Steady-state probabilities for each marking.
        token_probability_density_function: Token probability density function per place.
        average_tokens_per_place: Average number of tokens for each place.
        average_tokens_network: Average number of tokens for the whole network.
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

    def __init__(self, data: dict):
        # Ensure data exists and has expected keys before accessing
        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary.")
        required_keys = ['petri_net', 'arr_vlist', 'arr_edge', 'arr_tranidx',
                         'spn_labda', 'spn_steadypro', 'spn_markdens',
                         'spn_allmus', 'spn_mu']
        if not all(key in data for key in required_keys):
            missing = [key for key in required_keys if key not in data]
            raise ValueError(f"Missing required keys in input data: {missing}")

        self.spn = np.array(data['petri_net'])
        # Basic validation for spn shape
        if self.spn.ndim != 2 or self.spn.shape[1] % 2 != 1:
            print(f"Warning: Unexpected SPN shape {self.spn.shape}. Expected (places, 2*transitions + 1).")

        self.incidence_matrix = to_incidence_matrix(self.spn)
        self.reachability_graph_nodes = [np.array(marking) for marking in data['arr_vlist']]
        self.reachability_graph_edges = np.array(data['arr_edge'])
        self.transition_indices = np.array(data['arr_tranidx'])
        self.average_firing_rates = np.array(data['spn_labda'])
        self.steady_state_probabilities = np.array(data['spn_steadypro'])
        self.token_probability_density_function = np.array(data['spn_markdens'])
        self.average_tokens_per_place = np.array(data['spn_allmus'])
        self.average_tokens_network = data['spn_mu']

        # Basic shape validation for analysis results
        num_transitions = self.spn.shape[1] // 2
        num_places = self.spn.shape[0]
        if self.average_firing_rates.shape[0] != num_transitions:
            print(
                f"Warning: Mismatch in average_firing_rates shape {self.average_firing_rates.shape} and number of transitions {num_transitions}.")
        if self.average_tokens_per_place.shape[0] != num_places:
            print(
                f"Warning: Mismatch in average_tokens_per_place shape {self.average_tokens_per_place.shape} and number of places {num_places}.")

    def to_information(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the Petri net structure into graph information suitable for GNNs.

        Represents the Petri net as a homogeneous graph where nodes are
        either places or transitions.

        Node Features: Four features per node:
                       [Is_Place (1 if place, 0 if transition),
                        Is_Transition (1 if transition, 0 if place),
                        Initial Marking (for places, 0 for transitions),
                        Firing Rate (for transitions, 0 for places)].
                       Shape: (num_nodes, 4).

        Edges: Represent arcs P->T and T->P.
               Shape: (num_arcs, 2). Edges between nodes (source_node_idx, target_node_idx).

        Edge Features: One feature per edge: [Arc Weight].
                       Shape: (num_arcs, 1).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - node_features (np.ndarray): Shape (num_nodes, 4).
                - edge_features (np.ndarray): Shape (num_arcs, 1).
                - edge_pairs (np.ndarray): Shape (num_arcs, 2).
        """
        num_places = self.spn.shape[0]
        # Assuming spn has shape (num_places, 2 * num_transitions + 1)
        num_transitions = self.spn.shape[1] // 2
        num_nodes = num_places + num_transitions

        # Extract components from the SPN matrix
        pre_conditions = self.spn[:, :num_transitions]  # Input matrix I (P -> T)
        post_conditions = self.spn[:, num_transitions:2 * num_transitions]  # Output matrix O (T -> P)
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

    def get_analysis_result(self, label: SPNAnalysisResultLabel) -> Union[np.ndarray, float]:
        """
        Fetches a specific analysis result by its label.
        Args:
            label: The name of the analysis result to fetch. Must be one of
                   the values defined in `SPNAnalysisResultLabel`.

        Returns:
            The value of the requested analysis result, which can be either
            a NumPy array or a float, depending on the label.

        Raises:
            AttributeError: If the label is somehow invalid (though `Literal`
                            should prevent this at static analysis time for type checkers).
        """
        # Use getattr to dynamically access the attribute based on the label string
        if not hasattr(self, label):
            raise AttributeError(f"Invalid analysis result label: {label}")
        return getattr(self, label)


def to_incidence_matrix(pn: np.ndarray) -> np.ndarray:
    """Converts a Petri net representation [I; O; M_0] to its incidence matrix C = O - I.

    Args:
        pn: A NumPy array representing the Petri net [I; O; M_0]. It must be
            2D and have an odd number of columns.
            Shape: (num_places, 2 * num_transitions + 1).

    Returns:
        A NumPy array representing the incidence matrix (C).
        Shape: (num_places, num_transitions).
    """
    if pn.ndim != 2:
        raise ValueError("Input Petri net matrix must be 2-dimensional.")
    num_transitions = pn.shape[1] // 2
    if pn.shape[1] % 2 != 1:
        raise ValueError("Input Petri net matrix must have an odd number of columns (2*transitions + 1).")

    pre_conditions = pn[:, :num_transitions]
    post_conditions = pn[:, num_transitions: 2 * num_transitions]
    incidence_matrix = post_conditions - pre_conditions
    return incidence_matrix


def load_spn_data_from_files(
        file_paths: Union[Path, List[Path]]
) -> List[SPNData]:
    """
    Loads a list of SPNData objects from one or more JSON-L files.

    This function reads all data into memory and returns a list. For very large
    datasets, consider using `load_spn_data_lazily` to save memory.

    Args:
        file_paths: A single Path object or a list of Path objects pointing
                    to the JSON-L data files.

    Returns:
        A list of SPNData instances, one for each line in the provided files.
    """
    # This is a convenience wrapper around the lazy loader to materialize the list.
    return list(load_spn_data_lazily(file_paths))


def load_spn_data_lazily(
        file_paths: Union[Path, List[Path]]
) -> Iterator[SPNData]:
    """
    Lazily loads SPNData objects from one or more JSON-L files using a generator.

    This function is highly memory-efficient as it yields one SPNData object at
    a time, without storing the entire dataset in memory.

    Args:
        file_paths: A single Path object or a list of Path objects pointing
                    to the JSON-L data files.

    Yields:
        An iterator of SPNData instances.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
        json.JSONDecodeError: If a line in the file is not valid JSON.
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
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Each line is a separate JSON object
                data_dict = json.loads(line)
                # 'yield' turns this function into a generator
                yield SPNData(data_dict)
