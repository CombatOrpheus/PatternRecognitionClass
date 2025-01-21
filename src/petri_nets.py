import numpy as np

from typing import List, Tuple


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
    return pn[:, :transitions] - pn[:, transitions*2:-1]


def to_information(petri_net: np.ndarray, lambdas: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    num_transitions = petri_net.shape[1] // 2

    pre_conditions = petri_net[:, :num_transitions]
    post_conditions = petri_net[:, num_transitions:2 * num_transitions]
    initial_marking = petri_net[:, -1]

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
