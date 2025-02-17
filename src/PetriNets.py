from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SPNData:
    """
    Represents a Stochastic Petri net with its incidence matrix, output matrix,
    initial marking, and additional analysis results.

    Attributes:
        spn: The Stochastic Petri Net in compound form (column-based) [I; O; M_0].
        incidence_matrix: The incidence matrix.
        reachability_graph_nodes: List of markings in the reachability graph.
        reachability_graph_edges: Edges of the reachability graph.
        transition_indices: Transition indices that fired to generate each
                            marking in `reachability_graph_nodes`.
        average_firing_rates: Average firing rates (lambda) for each arc in
                              the reachability graph.
        steady_state_probabilities: Steady-state probabilities.
        token_probability_density_function: Token probability density function.
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

    def to_information(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a Petri net and associated parameters into graph information.

        This function processes a given Petri net matrix and its associated
        rates (lambdas) to generate graph-based representations. The output
        includes a set of edges with attributes, node features, and additional
        graph-related data.

        Parameters:
            petri_net (np.ndarray): A 2D NumPy array representing the Petri net
                structure. The matrix includes input (pre), output (post) arcs,
                and the initial marking.
            lambdas (np.ndarray): A 1D NumPy array representing the transition
                rates or weights for each transition in the Petri net.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three
            elements:
                - all_edges (np.ndarray): A 2D NumPy array of shape (num_edges, 2),
                  representing all edges in the graph, where the first column
                  contains source nodes and the second column contains target nodes.
                - edge_features (np.ndarray): A 2D NumPy array of shape
                  (num_edges, 3), where each row includes attributes of the
                  corresponding edge - [weight from pre-matrix, weight from
                  post-matrix, transition rate].
                - node_features (np.ndarray): A 1D NumPy array containing the
                  concatenation of the initial marking and transition rates
                  (lambdas), representing features of the nodes.
        """
        num_transitions = self.spn.shape[1] // 2

        pre_conditions = self.spn[:, :num_transitions]
        post_conditions = self.spn[:, num_transitions:2 * num_transitions]
        initial_marking = self.spn[:, -1]
        lambdas = self.average_firing_rates

        incoming_edges = np.transpose(np.nonzero(pre_conditions))
        outgoing_edges = np.transpose(np.nonzero(post_conditions))
        all_edges = np.vstack((incoming_edges, outgoing_edges))
        all_edges = np.flip(all_edges, axis=1)  # Flip so that it is source, target

        num_edges = all_edges.shape[0]
        edge_features = np.zeros((num_edges, 3))
        edge_features[:, 0] = pre_conditions[all_edges[:, 0], all_edges[:, 1]]
        edge_features[:, 1] = post_conditions[all_edges[:, 0], all_edges[:, 1]]
        edge_features[:, 2] = lambdas[all_edges[:, 1]]

        node_features = np.concatenate((initial_marking, lambdas))
        return all_edges, edge_features, node_features


def to_incidence_matrix(pn: np.array) -> np.array:
    """Converts a Petri net representation to its incidence matrix.  This version
    assumes an odd number of columns in the input array.

    Args:
        pn: A NumPy array representing the Petri net. The array should have an odd
            number of columns.  The first half (rounded down) represents pre-transitions,
            and the second half (rounded up) represents post-transitions.

    Returns:
        A NumPy array representing the incidence matrix.
    """
    transitions = pn.shape[1] // 2
    return pn[:, :transitions] - pn[:, transitions:-1]
