import numpy as np

from typing import *


def to_incidence_matrix(pn: np.array) -> np.array:
    _, num_transitions = pn.shape
    num_transitions //= 2
    return pn[:, :num_transitions] - pn[:, num_transitions:-1]


def place_transition_pairs(pn: np.array) -> List[Tuple[int, int]]:
    pass


def to_edge_list(pn: np.array) -> List[Tuple[int, int]]:
    pass
