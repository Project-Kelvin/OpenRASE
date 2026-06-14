"""
This defines the models used in the root evolver service/client of the HiGENESIS algorithm.
"""

class SelectNeighbour(dict):
    """
    Model of the request body for selecting a neighbour of the root individual.
    """

    root: int
    radius: float
