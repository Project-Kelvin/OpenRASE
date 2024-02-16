"""
Defines the TrafficGenerator class.
"""

from typing import Any

from shared.models.embedding_graph import EmbeddingGraph
from constants.notification import FORWARDING_GRAPH_DEPLOYED
from mano.notification_system import Subscriber


class TrafficGenerator(Subscriber):
    """
    Class that generates traffic.
    """

    _design = None
    _embeddingGraphs: "list[EmbeddingGraph]" = []

    def setDesign(self, design) -> None:
        """
        Set the design of the traffic generator.

        Parameters:
            design (dict): The design of the traffic generator.
        """

        self._design = design

    def _generateTraffic(self) -> None:
        """
        Generate traffic.
        """

        pass

    def getData(self) -> Any:
        """
        Get the data from the traffic generator.
        """

        pass

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == FORWARDING_GRAPH_DEPLOYED:
            fg: EmbeddingGraph = args[0]
            self._embeddingGraphs.append(fg)

            pass
