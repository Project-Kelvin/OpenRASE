"""
Defines the SFCRequestGenerator abstract class.
"""

from abc import ABC, abstractmethod
from mano.orchestrator import Orchestrator


class ISFCRequestGenerator(ABC):
    """
    Abstract class for generating SFC requests.
    """

    _orchestrator: Orchestrator = None

    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Constructor for the class.

        Parameters:
            orchestrator (Orchestrator): The orchestrator.
        """

        self._orchestrator = orchestrator

    @abstractmethod
    def generateRequests(self) -> None:
        """
        Generate SFC requests.
        """
