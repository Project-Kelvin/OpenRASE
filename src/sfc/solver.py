"""
Defines the abstract Solver class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from shared.models.sfc_request import SFCRequest
from sfc.traffic_generator import TrafficGenerator

if TYPE_CHECKING:
    from mano.orchestrator import Orchestrator


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
    def generateEmbeddingGraphs(self) -> None:
        """
        Generate Forwarding Graphs for the SFC requests.
        """
