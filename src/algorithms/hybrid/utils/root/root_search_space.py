"""
Defines the root search space class for the HiGENESIS algorithm.
"""

import datetime
import os
import random
from threading import Lock
import numpy as np
from shared.utils.config import getConfig
from algorithms.hybrid.utils.root.root_interface import RootInterface

artifacts: str = os.path.join(getConfig()["repoAbsolutePath"], "artifacts")
if not os.path.exists(artifacts):
    os.makedirs(artifacts)
experiments: str = os.path.join(artifacts, "experiments")
if not os.path.exists(experiments):
    os.makedirs(experiments)

rootDir: str = os.path.join(experiments, "root")
if not os.path.exists(rootDir):
    os.makedirs(rootDir)

currentRootExperimentFile: str = os.path.join(rootDir, f"experiment_{str(datetime.datetime.now())}.csv")

class RootSearchSpace(RootInterface):
    lock = Lock()

    def __init__(self, popSize: int):
        """
        Initializes the RootSearchSpace with the given population size.

        Parameters:
            popSize (int): The size of the population for the HiGENESIS algorithm.
        """

        self._popSize: int = popSize
        self._rootSearchSpace: list[int] = self._generateRootSearchSpace()
        self._rootFitness: dict[int, list[float]] = {}
        self._initializeRootFitness()
        self._generateLogHeader()

    def _generateLogHeader(self) -> None:
        """
        Generates the header for the root experiment log file.

        Returns:
            None
        """

        roots: list[str] = []

        for root in self._rootSearchSpace:
            roots.append(str(root))

        with open(currentRootExperimentFile, "w") as f:
            f.write(f"Timestamp,{','.join(roots)}\n")

    def _generateRootSearchSpace(self) -> list[int]:
        """
        Generates the search space for the root individual.

        Returns:
            list[int]: The list of possible values for the root individual.
        """

        return [i for i in range(1, self._popSize + 1) if self._popSize % i == 0]

    def _initializeRootFitness(self) -> None:
        """
        Initializes the fitness values for each root individual in the search space.

        Returns:
            None
        """

        for root in self._rootSearchSpace:
            self._rootFitness[root] = [1.0 / len(self._rootSearchSpace)]  # Initialize with a list of 10 fitness values

    def generateRandomRoot(self) -> int:
        """
        Generates a random root individual from the search space.

        Parameters:
            searchSpace (list[int]): The search space for the root individual.

        Returns:
            int: The randomly generated root individual.
        """

        weights: list[float] = [float(np.mean(self._rootFitness[root])) for root in self._rootSearchSpace]
        # root: int = random.choices(self._rootSearchSpace, weights=weights, k=1)[0]
        root: int = 1

        return root

    def setRootFitness(self, root: int, fitness: float) -> None:
        """
        Sets the fitness values for a given root individual.

        Parameters:
            root (int): The root individual.
            fitness (float): The fitness values to set.

        Returns:
            None
        """

        with RootSearchSpace.lock:
            if root not in self._rootFitness:
                self._rootFitness[root] = []
            self._rootFitness[root].append(fitness)

        with open(currentRootExperimentFile, "a") as f:
            fitnessValues: list[str] = [str(np.mean(self._rootFitness[root])) for root in self._rootSearchSpace]
            f.write(f"{str(datetime.datetime.now())},{','.join(fitnessValues)}\n")

    def selectNextRoot(self, currentRoot: int, radius: float) -> int:
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
            if abs(i - currentRootIndex) <= indexRadius and self._rootSearchSpace[i] != currentRoot
        ]

        if len(neighbours) == 0:

            neighbours = [root for root in self._rootSearchSpace if root != currentRoot]

        weights: list[float] = [float(np.mean(self._rootFitness[root])) for root in neighbours]

        newRoot: int = random.choices(neighbours, weights=weights, k=1)[0]

        return newRoot
