"""
This defines the root interface for the HiGENESIS algorithm.
"""

from abc import ABC, abstractmethod


class RootInterface(ABC):
    """
    Abstract base class for the root interface of the HiGENESIS algorithm.
    """

    @abstractmethod
    def generateRandomRoot(self) -> int:
        """
        Generates a random root individual from the search space.

        Returns:
            int: The randomly generated root individual.
        """

        pass

    @abstractmethod
    def setRootFitness(self, root: int, fitness: float) -> None:
        """
        Sets the fitness values for a given root individual.

        Parameters:
            root (int): The root individual.
            fitness (float): The fitness values to set.
        """

        pass

    @abstractmethod
    def selectNextRoot(self, currentRoot: int, radius: float) -> int:
        """
        Selects a neighbour for the current root individual.

        Parameters:
            currentRoot (int): The current root individual.
            radius (float): The radius for selecting the neighbour.
        """

        pass
