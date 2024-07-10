import json
import numpy
from pathlib import Path
from typing import Iterable, Dict

# This modules assumes that the files being used have the following structure:
# petri_net: A Petri Net in compound matrix form [I, O, M_0]; transitions are columns, while rows are places.
# arr_vlist: The set of markings on the reachability graph.
# arr_edge: The edges for the reachability graph.
# arr_tranidx: The transition that fired to generate the new marking.
# spn_labda: The lambda (average firing rate) corresponding to each arc of the reachable marking graph.
# spn_steadypro: The steady state probability.
# spn_markdens: The token probability density function.
# spn_mu: The average number of tokens.
#
# The original data is available in https://github.com/netlearningteam/SPN-Benchmark-DS
# Converted from a giant list into one JSON element per line using the command `jq -c '.[]' file`.


def get_data_line_iterator(file: Path) -> Iterable[Dict]:
    """Returns a a generator that yields Dicts containing Petri Nets. Assumes
    the JSON file has the following keys for each element:
        petri_net: A Petri Net in compound matrix form [I, O, M_0]; transitions are columns, while rows are places.
        arr_vlist: The set of markings on the reachability graph.
        arr_edge: The edges for the reachability graph.
        arr_tranidx: The transition that fired to generate the new marking.
    This method assumes that the file has one JSON element per line.
    """
    with open(file, 'rb') as source:
        for line in source:
            yield json.loads(line)


def get_petri_graph(pn: numpy.array):
    """Given a Petri Net generate its graph representation.
        Inputs:
            pn: A Petri Net represented as a compound matrix [A-, A+, M_0]
        Outputs:
            A 4-tuple containing:
                places: list
                transitions: list
                pt_edges: place -> transition pairs
                tp_edges: transition -> places pairs
    """
    num_places, num_transitions = pn.shape
    # The number of transitions is (columns-1)/2, so we can simply round down.
    num_transitions = num_transitions // 2
    places = list(range(num_places))
    transitions = (list(range(num_places, num_places+num_transitions)))

    # place -> transition
    # Find the edges and correct indices
    pt_edges = numpy.argwhere(pn[:, 0:num_transitions])
    pt_edges += numpy.array([0, num_places])
    # transition -> place
    # Find the edges, correct indices and swap columns
    tp_edges = numpy.argwhere(pn[:, num_transitions:-1])
    tp_edges += numpy.array([0, num_places])
    tp_edges[:, [0, 1]] = tp_edges[:, [1, 0]]

    return (places, transitions, pt_edges, tp_edges)


def get_petri_nets(source: Iterable) -> Iterable:
    for data in source:
        pn = numpy.array(data['petri_net'])
        reachable_markings = numpy.array(data['arr_vlist'])
        edges = numpy.array(data['arr_edge'])
        fired_transitions = numpy.array(data['arr_tranidx'])
        yield (get_petri_graph(pn), (reachable_markings, edges, fired_transitions))


# Returning two tuples of values is not great, but it could be worse. This is sufficient for my needs at the moment and should be reasonably efficient.
