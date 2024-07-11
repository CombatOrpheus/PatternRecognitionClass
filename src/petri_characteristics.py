import numpy
from typing import List


def is_safe(reachability: numpy.array) -> bool:
    return numpy.all(reachability <= 0)


def is_strictly_conservative(reachability: numpy.array) -> bool:
    sums = numpy.sum(reachability, axis=1)
    return numpy.all(sums[:, 0] == sums)


def is_conservative(reachability: numpy.array) -> bool:
    pass


def is_live(reachability_edges: numpy.array, transitions: List[int]) -> bool:
    """Given a Reachability Graph, determine if it is live. For this particular
    function, liveness is determined as no transition being `level 0` live.
    """
    result = numpy.isin(reachability_edges.ravel(), transitions)
    return numpy.all(result)
