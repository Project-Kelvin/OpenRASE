""""
This defines the client for the root node of the HiGENESIS algorithm.
"""

import requests

from algorithms.hybrid.constants.root_evolver import GENERATE_RANDOM_ROOT_PATH, ROOT_PATH, SELECT_NEIGHBOUR_PATH, SERVICE_ADDRESS, SERVICE_PORT, SET_ROOT_FITNESS_PATH
from algorithms.hybrid.models.root_evolver import InitRootSearchSpace, SelectNeighbour, SetRootFitness
from algorithms.hybrid.utils.root.root_interface import RootInterface


class RootClient(RootInterface):
    def __init__(self):
        """
        Initializes the RootClient with the given root service URL.

        Parameters:
            rootServiceUrl (str): The URL of the root service.
        """

        self._rootServiceUrl: str = f"{SERVICE_ADDRESS}:{SERVICE_PORT}"

    def generateRandomRoot(self) -> int:
        """
        Generates a random root individual using the root service.

        Returns:
            int: The newly generated root individual.
        """

        response = requests.post(f"{self._rootServiceUrl}{GENERATE_RANDOM_ROOT_PATH}")
        response.raise_for_status()

        return response.json()

    def selectNextRoot(self, currentRoot: int, radius: float) -> int:
        """
        Selects a neighbour for the current root individual in the root service.

        Parameters:
            currentRoot (int): The current root individual.
            radius (float): The radius for selecting the neighbour.

        Returns:
            None
        """

        payload: SelectNeighbour = SelectNeighbour(root=currentRoot, radius=radius)
        response = requests.post(f"{self._rootServiceUrl}{SELECT_NEIGHBOUR_PATH}", json=payload)
        response.raise_for_status()

        root: int = response.json()

        return root

    def setRootFitness(self, root: int, fitness: float) -> None:
        """
        Sets the fitness of the current root individual in the root service.

        Parameters:
            root (int): The current root individual.
            fitness (float): The fitness value to set.

        Returns:
            None
        """

        payload: SetRootFitness = SetRootFitness(root=root, fitness=fitness)
        response = requests.post(f"{self._rootServiceUrl}{SET_ROOT_FITNESS_PATH}", json=payload)
        response.raise_for_status()

    def init(self, popSize: int) -> None:
        """
        Initializes the root service with the given population size.

        Parameters:
            popSize (int): The population size for the root service.

        Returns:
            None
        """

        payload: InitRootSearchSpace = InitRootSearchSpace(popSize=popSize)
        response = requests.post(f"{self._rootServiceUrl}{ROOT_PATH}", json=payload)
        response.raise_for_status()
