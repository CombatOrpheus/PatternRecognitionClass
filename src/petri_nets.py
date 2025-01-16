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


def to_information(petri_net: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a Petri net matrix to its graph representation, extracting edges, node
    features, and edge features for further use in graph-based tasks or analyses.

    Parameters
    ----------
    petri_net : numpy.ndarray
        A 2D numpy array representing the Petri net structure. Columns are divided
        into pre-transitions, post-transitions, and the initial marking.

    Returns
    -------
    tuple
        A tuple containing three elements:
        - all_edges (numpy.ndarray): A 2D array where each row represents an edge
          in the graph. Each edge connects a node pair specified by the column
          indices (source, destination).
        - edge_features (numpy.ndarray): A 2D array of features extracted for all
          edges. Each feature corresponds to the weight or marked connections in
          pre and post-transitions.
        - node_features (numpy.ndarray): A 1D array of features extracted for all
          nodes. Summarizes per-node attributes or markings within the Petri net.
    """
    num_transitions = petri_net.shape[1] // 2

    pre_transitions = petri_net[:, :num_transitions]
    post_transitions = petri_net[:, num_transitions:2 * num_transitions]

    # Vectorized edge detection
    incoming_edges = np.transpose(np.nonzero(pre_transitions))
    outgoing_edges = np.transpose(np.nonzero(post_transitions))
    all_edges = np.vstack((incoming_edges, outgoing_edges))
    all_edges = all_edges[:, [1, 0]]

    incoming_edge_features = pre_transitions[incoming_edges]
    outgoing_edge_features = post_transitions[outgoing_edges]

    node_features = np.hstack((petri_net[:, -1].transpose(), np.full(num_transitions, -1)))
    edge_features = np.vstack((incoming_edge_features, outgoing_edge_features))

    return all_edges, edge_features, node_features