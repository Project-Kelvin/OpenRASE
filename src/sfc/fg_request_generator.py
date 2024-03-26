"""
Defines the FGRequestGenerator abstract class.
"""

from abc import ABC, abstractmethod
from mano.orchestrator import Orchestrator


class FGRequestGenerator(ABC):
    """
    Abstract class for generating FG requests.
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
        Generate FG requests.
        """
