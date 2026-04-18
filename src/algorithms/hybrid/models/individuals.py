"""
Defines the classes for GA individuals.
"""

from uuid import UUID, uuid4
from deap import base


class Fitness(base.Fitness):
    """
    Class responsible for the fitness of the hierarchical evolution.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the fitness.

        Returns:
            None
        """

        self.weights = (1.0, -1.0)
        super().__init__(*args, **kwargs)


class Individual(list):
    """
    Individual class for DEAP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id: UUID = uuid4()
        self.fitness: base.Fitness = Fitness()


class GenesisIndividual(Individual):
    """
    Class representing an individual in the GENESIS algorithm.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the individual.

        Parameters:
            weights (list[float]): the weights of the individual.

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        self._metaIndividual: Individual = Individual([0.0, 0.0])

    @property
    def metaIndividual(self) -> Individual:
        """
        Gets the meta-individual.

        Returns:
            Individual: the meta-individual.
        """

        return self._metaIndividual

    @metaIndividual.setter
    def metaIndividual(self, value: Individual) -> None:
        """
        Sets the meta-individual.

        Parameters:
            value (Individual): the new meta-individual.

        Returns:
            None
        """

        self._metaIndividual = value
