"""
Defines the SFCRequestGenerator abstract class.
"""

from abc import ABC, abstractmethod
from sfc.solver import Solver


class SFCRequestGenerator(ABC):
    """
    Abstract class for generating SFC requests.
    """

    _solver: Solver = None

    def __init__(self, solver: Solver) -> None:
        """
        Constructor for the class.

        Parameters:
            solver (Solver): The solver.
        """

        self._solver = solver

    @abstractmethod
    def generateRequests(self) -> None:
        """
        Generate SFC requests.
        """
