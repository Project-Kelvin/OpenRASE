"""
Defines the abstractEVolver class.
"""

from abc import ABC, abstractmethod

from shared.models.sfc_request import SFCRequest
from manager.orchestrator import Orchestrator
from manager.traffic_generator import TrafficGenerator

class Evolver(ABC):
    """
    Abstract class that defines the interface for the evolver.
    """

    _sfcRequests: "list[SFCRequest]" = []
    _trafficGenerator: TrafficGenerator = None
    _orchestrator: Orchestrator = None

    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._trafficGenerator = trafficGenerator

    @abstractmethod
    def generateForwardingGraphs(self, sfcRequests: "list[SFCRequest]") -> None:
        """
        Generate Forwarding Graphs for the SFC requests.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """

        self._sfcRequests.append(sfcRequests)

        pass
