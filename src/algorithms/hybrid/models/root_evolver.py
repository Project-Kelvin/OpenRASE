"""
This defines the models used in the root evolver service/client of the HiGENESIS algorithm.
"""

class SelectNeighbour(dict):
    """
    Model of the request body for selecting a neighbour of the root individual.
    """

    root: int
    radius: float

class SetRootFitness(dict):
    """
    Model of the request body for setting the fitness of the root individual.
    """

    root: int
    fitness: float

class InitRootSearchSpace(dict):
    """
    Model of the request body for initializing the root search space.
    """

    popSize: int
