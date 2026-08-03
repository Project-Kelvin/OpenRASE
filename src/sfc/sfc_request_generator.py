"""
Defines the SFCRequestGenerator abstract class.
"""

from abc import ABC, abstractmethod
from mano.orchestrator import Orchestrator


class SFCRequestGenerator(ABC):
    """
    Abstract class for generating SFC requests.
    """


    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Constructor for the class.

        Parameters:
            orchestrator (Orchestrator): The orchestrator.
        """

        self._orchestrator: Orchestrator = orchestrator

    @abstractmethod
    def generateRequests(self) -> None:
        """
        Generate SFC requests.
        """
