"""
Defines the abstract Solver class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from sfc.traffic_generator import TrafficGenerator

if TYPE_CHECKING:
    from mano.orchestrator import Orchestrator


class Solver(ABC):
    """
    Abstract class that defines the interface for the Solver.
    """

    _requests: "list[Union[SFCRequest, EmbeddingGraph]]" = []
    _trafficGenerator: TrafficGenerator = None
    _orchestrator: Orchestrator = None

    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._trafficGenerator = trafficGenerator

    def sendRequests(self, requests: "list[Union[SFCRequest, EmbeddingGraph]]") -> None:
        """
        Send the SFC requests to the orchestrator.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """

        self._requests.append(requests)

    @abstractmethod
    def generateEmbeddingGraphs(self) -> None:
        """
        Generate Forwarding Graphs for the SFC requests.
        """
