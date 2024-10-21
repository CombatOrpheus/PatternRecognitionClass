import json
from pathlib import Path
from typing import Iterable

import numpy as np

# This modules assumes that the files being used have the following structure:
# petri_net: A Petri Net in compound matrix form [I, O, M_0]; transitions are
# columns, while rows are places.
# arr_vlist: The set of markings on the reachability graph.
# arr_edge: The edges for the reachability graph.
# arr_tranidx: The transition that fired to generate the new marking.
# spn_labda: The lambda (average firing rate) corresponding to each arc of the
# reachable marking graph.
# spn_steadypro: The steady state probability.
# spn_markdens: The token probability density function.
# spn_all_mus: The average number of tokens for each place.
# spn_mu: The average number of tokens for the whole networ.
#
# The original data is available in https://github.com/netlearningteam/SPN-Benchmark-DS
# Converted from a giant list into one JSON element per line using the command
# `jq -c '.[]' file`.


def get_petri_graph(pn: np.array):
    """Convert a Petri Net matrix into graph information
    Parameters
    ----------
        pn: A Petri Net represented as a compound matrix [A-, A+, M_0]
    Returns
    -------
        A 6-tuple containing:
            The list of places
            The list of transitions
            Place -> transition pairs
            Transition -> places pairs
            Initial marking of the Petri Net
            Petri Net incidence matrix (A_in - A_out)
    """
    num_places, num_transitions = pn.shape
    num_transitions = num_transitions // 2
    places = list(range(num_places))
    transitions = list(range(num_places, num_places + num_transitions))

    # place -> transition
    # Find the edges and correct indices
    A_in = pn[:, 0:num_transitions]
    pt_edges = np.argwhere(A_in)
    pt_edges += np.array([0, num_places])

    # transition -> place
    # Find the edges, correct indices and swap columns
    A_out = pn[:, num_transitions:-1]
    tp_edges = np.argwhere(A_out)
    tp_edges += np.array([0, num_places])
    tp_edges[:, [0, 1]] = tp_edges[:, [1, 0]]

    return (places, transitions, pt_edges, tp_edges, pn[:, -1], A_out - A_in)


def get_data(source_file: Path) -> Iterable:
    """
    Parameters
    ----------
        source_file: A valid Path
    """
    with open(source_file) as f:
        for data in map(json.loads, f):
            elements = list(data.values())
            petri_net = np.array(elements[0])
            reachability_graph = list(map(np.array, elements[1:-1]))
            spn_mu = elements[-1]
            yield get_petri_graph(petri_net), reachability_graph, spn_mu


def get_petri_nets(source_file: Path) -> Iterable:
    with open(source_file) as f:
        for data in map(json.loads, f):
            net = np.array(data["petri_net"])
            yield get_petri_graph(net)


def get_average_tokens(source: Path, network: bool = True) -> Iterable:
    """Return an iterator with the reachability graphs for each Petri Net in
    the source file.
    Parameters
    ----------
        source: A Path to the file
        network: Whether to return the average number of tokens for the whole network (True) or for each place (False).
    Returns
    -------
        iterable: An iterable that yields the graph information
        and the true label for this graph.
    """
    with open(source) as f:
        source = map(json.loads, f)
        for elem in source:
            data = list(elem.values())
            graph_data = [np.array(info) for info in data[1:5]]
            graph_data[0] = get_petri_graph(graph_data[0])
            label = data[-1] if network else data[-2]
            yield graph_data, label


def get_steady_state(source: Path) -> Iterable:
    with open(source) as f:
        source = map(json.loads, f)
        for elem in source:
            data = list(elem.values())
            graph_data = [np.array(info) for info in data[1:4]]
            graph_data[0] = get_petri_graph(graph_data[0])
            label = data[5]
            yield graph_data, label
