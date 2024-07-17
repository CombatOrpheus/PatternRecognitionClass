import numpy
from typing import List


def is_safe(reachability: numpy.array) -> bool:
    return numpy.all(reachability <= 1)


def is_strictly_conservative(reachability: numpy.array) -> bool:
    """Determine if the number of tokens in the Petri Net remains constant for
    all markings.
    """
    sums = numpy.sum(reachability, axis=1)
    return numpy.all(sums[:, 0] == sums)


def is_conservative(reachability: numpy.array) -> numpy.array:
    """If the Petri Net is convservative `with respect to a weigthing vector`,
    return this weighting vector; otherwise, return an empty vector.
    """
    pass


def is_live(reachability_edges: numpy.array, transitions: List[int]) -> bool:
    """Given a Reachability Graph, determine if it is live. For this particular
    function, liveness is determined as no transition being `level 0` live.
    """
    result = numpy.isin(reachability_edges.ravel(), transitions)
    return numpy.all(result)
