"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from typing import Callable
from deap import tools
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from algorithms.ga_dijkstra_algorithm.ga_utils import (
    decodePop,
    generateRandomIndividual,
    mutate,
)
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator

POP_SIZE: int = 100

hybridEvolution: HybridEvolution = HybridEvolution(
    "ga_hybrid",
    decodePop,
    generateRandomIndividual,
    tools.cxTwoPoint,
    mutate
)

def solve(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: "list[TrafficDesign]",
    trafficGenerator: TrafficGenerator,
    telemetry: Telemetry,
    experiment: str
) -> None:
    """
    Solves the problem using a GA for VNF embedding and Dijkstra for link embedding.

    Parameters:
        topology (Topology): the topology to use for solving.
        fgrs (list[EmbeddingGraph]): the list of Forwarding Graphs to embed.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (list[TrafficDesign]): the traffic design to use for solving.
        trafficGenerator (TrafficGenerator): the traffic generator to use for solving.
        telemetry (Telemetry): telemetry instance.
        experiment (str): the experiment name.

    Returns:
        None
    """

    hybridEvolution.hybridSolve(
        topology,
        fgrs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        telemetry,
        POP_SIZE,
        experiment
    )
