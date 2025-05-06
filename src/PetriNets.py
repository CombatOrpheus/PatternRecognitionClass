from dataclasses import dataclass
from typing import List, Tuple  # Added Tuple

import numpy as np


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
        # Clarification: Assuming average_firing_rates corresponds to transitions,
        # matching the columns of I/O matrices.
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
        self.spn = np.array(data['petri_net'])
        self.incidence_matrix = to_incidence_matrix(self.spn)
        self.reachability_graph_nodes = [np.array(marking) for marking in data['arr_vlist']]
        self.reachability_graph_edges = np.array(data['arr_edge'])
        self.transition_indices = np.array(data['arr_tranidx'])
        self.average_firing_rates = np.array(data['spn_labda'])
        self.steady_state_probabilities = np.array(data['spn_steadypro'])
        self.token_probability_density_function = np.array(data['spn_markdens'])
        self.average_tokens_per_place = np.array(data['spn_allmus'])
        self.average_tokens_network = data['spn_mu']

    def to_information(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert the Petri net structure into graph information suitable for GNNs.

        Represents the Petri net as a bipartite graph:
        Nodes: Places (indices 0 to num_places-1) and
               Transitions (indices num_places to num_nodes-1).
        Edges: Represent arcs P->T and T->P.
        Node Features: Two features per node:
                       [Initial Marking (-1 for transitions), Firing Rate (0 for places)].
        Edge Features: One feature per edge: [Arc Weight].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - edge_pairs (np.ndarray): Shape (num_arcs, 2). Edges between
                  nodes (source_node_idx, target_node_idx).
                - edge_features (np.ndarray): Shape (num_arcs, 1). Arc weights.
                - node_features (np.ndarray): Shape (num_nodes, 2). Node features
                  as described above ([Initial Marking, Firing Rate]).
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
        # Initialize node_features array (num_nodes, 2 features)
        # Use float to accommodate rates and -1 placeholder
        node_features = np.zeros((num_nodes, 2), dtype=float)

        # Feature 1: Initial Marking (Column 0)
        node_features[0:num_places, 0] = initial_marking  # Assign M_0 to places
        node_features[num_places:, 0] = -1.0  # Assign -1 to transitions

        # Feature 2: Firing Rate (Column 1)
        node_features[0:num_places, 1] = 0.0  # Assign 0 to places
        node_features[num_places:, 1] = lambdas  # Assign rates to transitions

        # --- Edges and Edge Features (Bipartite Graph Structure) ---
        # Define node indices for clarity
        place_node_indices = np.arange(num_places)
        # Offset transition indices to follow place indices
        transition_node_indices = np.arange(num_places, num_nodes)

        # 1. Edges from Places to Transitions (from pre_conditions / Input matrix I)
        p_in_idx, t_in_idx = np.nonzero(pre_conditions)
        # Source nodes are places, target nodes are transitions (with offset)
        src_in = place_node_indices[p_in_idx]
        tgt_in = transition_node_indices[t_in_idx]
        edges_in = np.stack([src_in, tgt_in], axis=1)
        # Edge features are the weights from the pre_conditions matrix
        weights_in = pre_conditions[p_in_idx, t_in_idx]

        # 2. Edges from Transitions to Places (from post_conditions / Output matrix O)
        p_out_idx, t_out_idx = np.nonzero(post_conditions)
        # Source nodes are transitions (with offset), target nodes are places
        src_out = transition_node_indices[t_out_idx]
        tgt_out = place_node_indices[p_out_idx]
        edges_out = np.stack([src_out, tgt_out], axis=1)
        # Edge features are the weights from the post_conditions matrix
        weights_out = post_conditions[p_out_idx, t_out_idx]

        # Combine edges and features, handling cases where one set might be empty
        if edges_in.size > 0 and edges_out.size > 0:
            edge_pairs = np.vstack((edges_in, edges_out))
            # Ensure weights are concatenated in the same order as edges
            edge_features = np.concatenate((weights_in, weights_out)).reshape(-1, 1)
        elif edges_in.size > 0:  # Only incoming edges exist
            edge_pairs = edges_in
            edge_features = weights_in.reshape(-1, 1)
        elif edges_out.size > 0:  # Only outgoing edges exist
            edge_pairs = edges_out
            edge_features = weights_out.reshape(-1, 1)
        else:  # No edges
            edge_pairs = np.empty((0, 2), dtype=int)
            edge_features = np.empty((0, 1), dtype=float)  # Match weight type

        # Ensure correct shapes (optional but good practice for debugging)
        # num_arcs = edge_pairs.shape[0]
        # assert edge_pairs.shape == (num_arcs, 2), f"Shape mismatch for edge_pairs: {edge_pairs.shape}"
        # assert edge_features.shape == (num_arcs, 1), f"Shape mismatch for edge_features: {edge_features.shape}"
        # assert node_features.shape == (num_nodes, 2), f"Shape mismatch for node_features: {node_features.shape}"

        return node_features, edge_features, edge_pairs


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
    # Assuming pn is always correctly formatted (2D, odd columns)
    num_transitions = pn.shape[1] // 2
    # Extract Pre (I) and Post (O) matrices
    pre_conditions = pn[:, :num_transitions]
    post_conditions = pn[:, num_transitions: 2 * num_transitions]

    # Calculate Incidence Matrix C = Post - Pre
    incidence_matrix = post_conditions - pre_conditions
    return incidence_matrix
