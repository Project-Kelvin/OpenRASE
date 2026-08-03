"""
This defines the root evolver for the HiGENESIS algorithm.
"""

from typing import cast

from algorithms.hybrid.utils.root.root_client import RootClient
from algorithms.hybrid.utils.root.root_individual import RootIndividual
from algorithms.hybrid.utils.root.root_interface import RootInterface
from algorithms.hybrid.utils.root.root_search_space import RootSearchSpace


class RootEvolver:
    """
    The RootEvolver class is responsible for evolving the root individual in the HiGENESIS algorithm.
    It provides methods to generate a random root, select a neighbour, and set the fitness of the root individual.
    """

    _rootIndividual: RootIndividual = RootIndividual()
    def __init__(self, popSize: int, isClientMode: bool = False):
        """
        Initializes the RootEvolver with the given mode (client or server).

        Parameters:
            popSize (int): The population size for the root search space.
            isClientMode (bool): If True, the RootEvolver operates in client mode; otherwise, it operates in server mode.
        """

        self._rootInterface: RootInterface = RootClient() if isClientMode else RootSearchSpace(popSize)

        if isClientMode:
            cast(RootClient, self._rootInterface).init(popSize)

    def generateRandomRoot(self) -> int:
        """
        Generates a random root individual.

        Returns:
            int: The newly generated root individual.
        """

        root: int = self._rootInterface.generateRandomRoot()
        RootEvolver._rootIndividual.setRoot(root)

        return root

    @classmethod
    def getRoot(cls) -> int:
        """
        Returns the root.

        Returns:
            int: the root.
        """

        return RootEvolver._rootIndividual.getRoot()

    @classmethod
    def setRoot(cls, root: int) -> None:
        """
        Sets the root.

        Parameters:
            root (int): The new root value to set.
        """

        RootEvolver._rootIndividual.setRoot(root)

    def setRootFitness(self, root: int, fitness: float) -> None:
        """
        Sets the fitness of the current root individual.

        Parameters:
            root (int): The current root individual.
            fitness (float): The fitness value to set.

        Returns:
            None
        """

        self._rootInterface.setRootFitness(root, fitness)

    def selectNextRoot(self, currentRoot: int, radius: float) -> int:
        """
        Selects a neighbour for the current root individual.

        Parameters:
            currentRoot (int): The current root individual.
            radius (float): The radius for selecting the neighbour.

        Returns:
            int: The selected neighbour root individual.
        """

        root: int = self._rootInterface.selectNextRoot(currentRoot, radius)
        RootEvolver._rootIndividual.setRoot(root)

        return root
