"""
Defines the abstract Solver class.
"""

from abc import ABC, abstractmethod

from shared.models.sfc_request import SFCRequest
from mano.orchestrator import Orchestrator
from sfc.traffic_generator import TrafficGenerator


class Solver(ABC):
    """
    Abstract class that defines the interface for the Solver.
    """

    _sfcRequests: "list[SFCRequest]" = []
    _trafficGenerator: TrafficGenerator = None
    _orchestrator: Orchestrator = None

    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._trafficGenerator = trafficGenerator

    def sendSFCRequests(self, sfcRequests: "list[SFCRequest]") -> None:
        """
        Send the SFC requests to the orchestrator.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """

        self._sfcRequests.append(sfcRequests)

    @abstractmethod
    def generateEmbeddingGraphs(self, sfcRequests: "list[SFCRequest]") -> None:
        """
        Generate Forwarding Graphs for the SFC requests.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """
