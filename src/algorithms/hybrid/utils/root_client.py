""""
This defines the client for the root node of the HiGENESIS algorithm.
"""

import requests

from algorithms.hybrid.constants.root_evolver import SERVICE_ADDRESS, SERVICE_PORT
from algorithms.hybrid.models.root_evolver import SelectNeighbour


class RootClient:
    def __init__(self):
        """
        Initializes the RootClient with the given root service URL.

        Parameters:
            rootServiceUrl (str): The URL of the root service.
        """

        self._rootServiceUrl: str = f"{SERVICE_ADDRESS}:{SERVICE_PORT}"

    def getRootIndividual(self) -> int:
        """
        Retrieves the current root individual from the root service.

        Returns:
            int: The current root individual.
        """

        response = requests.get(f"{self._rootServiceUrl}/root")
        response.raise_for_status()

        return response.json()

    def selectRootNeighbour(self, root: int, radius: float) -> None:
        """
        Selects a neighbour for the current root individual in the root service.

        Parameters:
            root (int): The current root individual.
            radius (float): The radius for selecting the neighbour.

        Returns:
            None
        """

        payload: SelectNeighbour = SelectNeighbour(root=root, radius=radius)
        response = requests.post(f"{self._rootServiceUrl}/root-neighbour", json=payload)
        response.raise_for_status()
