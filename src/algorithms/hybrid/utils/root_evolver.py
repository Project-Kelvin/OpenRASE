"""
Defines the root evolver class for the HiGENESIS algorithm.
"""

import random
from threading import Lock

class RootEvolver:
    _rootIndividual: int = -1
    _exploredRoots: list[int] = []
    lock = Lock()

    def __init__(self, popSize: int):
        """
        Initializes the RootEvolver with the given population size.

        Parameters:
            popSize (int): The size of the population for the HiGENESIS algorithm.
        """

        self._popSize: int = popSize
        self._rootSearchSpace: list[int] = self.generateRootSearchSpace()

    @staticmethod
    def getRootIndividual() -> int:
        """
        Returns the current root individual.

        Returns:
            int: The current root individual.
        """

        return RootEvolver._rootIndividual

    @staticmethod
    def setRootIndividual(root: int) -> None:
        """
        Sets the current root individual.

        Parameters:
            root (int): The new root individual to set.

        Returns:
            None
        """

        with RootEvolver.lock:
            RootEvolver._rootIndividual = root

    def generateRootSearchSpace(self) -> list[int]:
        """
        Generates the search space for the root individual.

        Returns:
            list[int]: The list of possible values for the root individual.
        """

        return [i for i in range(1, self._popSize + 1) if self._popSize % i == 0]

    def generateRandomRoot(self) -> int:
        """
        Generates a random root individual from the search space.

        Parameters:
            searchSpace (list[int]): The search space for the root individual.

        Returns:
            int: The randomly generated root individual.
        """

        # return random.choice(self._rootSearchSpace)
        return 1

    def selectRootNeighbour(self, currentRoot: int, radius: float) -> int:
        """
        Selects a neighbour of the current root individual within a given radius.

        Parameters:
            currentRoot (int): The current root individual.
            radius (float): The radius within which to select a neighbour.

        Returns:
            int: The selected neighbour root individual.
        """

        currentRootIndex: int = self._rootSearchSpace.index(currentRoot)
        indexRadius: int = int(radius * len(self._rootSearchSpace)) if radius * len(self._rootSearchSpace) >= 1 else 1
        neighbours: list[int] = [
            self._rootSearchSpace[i]
            for i in range(0, len(self._rootSearchSpace))
            if abs(i - currentRootIndex) <= indexRadius and self._rootSearchSpace[i] != currentRoot and self._rootSearchSpace[i] not in RootEvolver._exploredRoots
        ]

        # if len(neighbours) == 0:
        #     if currentRootIndex == 0:
        #         if self._rootSearchSpace[currentRootIndex + 1] not in RootEvolver._exploredRoots:
        #             neighbours = [self._rootSearchSpace[currentRootIndex + 1]]
        #     elif currentRootIndex == len(self._rootSearchSpace) - 1:
        #         if self._rootSearchSpace[currentRootIndex - 1] not in RootEvolver._exploredRoots:
        #             neighbours = [self._rootSearchSpace[currentRootIndex - 1]]
        #     else:
        #         if self._rootSearchSpace[currentRootIndex + 1] not in RootEvolver._exploredRoots:
        #             neighbours.append(self._rootSearchSpace[currentRootIndex + 1])
        #         if self._rootSearchSpace[currentRootIndex - 1] not in RootEvolver._exploredRoots:
        #             neighbours.append(self._rootSearchSpace[currentRootIndex - 1])

        if len(neighbours) == 0:
            if len(RootEvolver._exploredRoots) == len(self._rootSearchSpace):
                with RootEvolver.lock:
                    RootEvolver._exploredRoots.clear()

            return random.choice([root for root in self._rootSearchSpace if root not in RootEvolver._exploredRoots])

        return random.choice(neighbours)

    def addRootToExploredRoot(self, root: int) -> None:
        """
        Adds a root individual to the list of explored roots.

        Parameters:
            root (int): The root individual to add.

        Returns:
            None
        """

        with RootEvolver.lock:
            if root not in RootEvolver._exploredRoots:
                RootEvolver._exploredRoots.append(root)
