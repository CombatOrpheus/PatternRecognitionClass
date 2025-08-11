import json
from pathlib import Path
from typing import Iterable, Tuple, List

import numpy as np


# This module processes data from the SPN-Benchmark-DS dataset
# (https://github.com/netlearningteam/SPN-Benchmark-DS).  The dataset
# consists of JSON files, where each line is a separate JSON object with the
# following structure:
#
# - petri_net: A Petri net represented as a compound matrix [I, O, M_0].
#   - I: Incidence matrix (transitions are columns, places are rows).
#   - O: Output matrix.
#   - M_0: Initial marking.
# - arr_vlist:  List of markings in the reachability graph. Each marking is a NumPy array.
# - arr_edge: Edges of the reachability graph.
# - arr_tranidx: Transition indices that fired to generate each marking in arr_vlist.
# - spn_labda: Average firing rates (lambda) for each arc in the reachability graph.
# - spn_steadypro: Steady-state probabilities.
# - spn_markdens: Token probability density function.
# - spn_all_mus: Average number of tokens for each place.
# - spn_mu: Average number of tokens for the whole network.
#
# The original data was converted from a single giant JSON list into one JSON
# object per line using the command `jq -c '.[]' file`.


def process_spn_data(
    source: Path, data_fields: List[int], label_field: int
) -> Iterable[Tuple[List[np.ndarray], np.ndarray]]:
    """
    Processes SPN data from a JSON file, extracting specified data fields and a label.

    Parameters
    ----------
    source : Path
        Path to the JSON data file.  Each line should be a valid JSON object.
    data_fields : List[int]
        List of indices indicating which fields to extract as NumPy arrays.
    label_field : int
        Index indicating which field to extract as the label.

    Returns
    -------
    Iterable[Tuple[List[np.ndarray], np.ndarray]]
        An iterable yielding tuples of (data, label).  Data is a list of NumPy arrays.
    """
    with open(source) as f:
        for line in f:
            data = json.loads(line)
            elements = list(data.values())
            graph_data = [np.array(elements[i]) for i in data_fields]
            label = elements[label_field]
            yield graph_data, label


def get_average_tokens(source: Path, network: bool = True) -> Iterable[Tuple[List[np.ndarray], np.ndarray]]:
    """Returns an iterator with reachability graphs and average token counts."""
    data_fields = list(range(1, 5))  # Fields 1-4 are graph data.
    label_field = -1 if network else -2
    return process_spn_data(source, data_fields, label_field)


def get_steady_state(source: Path) -> Iterable[Tuple[List[np.ndarray], np.ndarray]]:
    """Returns an iterator with reachability graphs and steady state probabilities."""
    return process_spn_data(source, list(range(1, 4)), 5)


def get_mark_density(source: Path) -> Iterable[Tuple[List[np.ndarray], np.ndarray]]:
    """Returns an iterator with reachability graphs and token probability density functions."""
    return process_spn_data(source, list(range(1, 4)), 6)
