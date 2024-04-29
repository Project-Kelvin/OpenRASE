"""
Defines the abstract Solver class.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from queue import Queue
from typing import TYPE_CHECKING, Union
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

if TYPE_CHECKING:
    from mano.orchestrator import Orchestrator


class Solver(ABC):
    """
    Abstract class that defines the interface for the Solver.
    """


    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__()
        self._orchestrator: Orchestrator = orchestrator
        self._trafficGenerator: TrafficGenerator = trafficGenerator
        self._requests: Queue = Queue()

    def sendRequests(self, requests: "list[Union[SFCRequest, EmbeddingGraph]]") -> None:
        """
        Send the SFC requests to the orchestrator.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """

        TUI.appendToLog(f"Receiving {len(requests)} requests:")
        for request in requests:
            TUI.appendToLog(f"  {request['sfcrID']}")

        for request in requests:
            self._requests.put(request)

    @abstractmethod
    def generateEmbeddingGraphs(self) -> None:
        """
        Generate Forwarding Graphs for the SFC requests.
        """
