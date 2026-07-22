"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from typing import Callable, Type
from deap import tools
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution, Individual
from algorithms.models.embedding import DecodedIndividual
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator
from algorithms.mak_ga.mak_ga_utils import MakGAUtils

POP_SIZE: int = 20
INDPB: float = 0.7 # Experimentally determined gene mutation probability for the GA
MUTPB: float = 0.7 # Experimentally determined mutation probability for the GA
CXPB: float = 1.0

def solve(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: "list[TrafficDesign]",
    trafficGenerator: TrafficGenerator,
    telemetry: Telemetry,
    experiment: str,
    mutPb: float = MUTPB,
    cxPb: float = CXPB,
    indPb: float = INDPB,
    evaluateOnline: bool = True,
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
        mutPb (float): the mutation probability.
        cxPb (float): the crossover probability.
        indPb (float): the individual mutation probability.
        evaluateOnline (bool): whether to evaluate the solution online or offline.

    Returns:
        None
    """

    gahaUtils: MakGAUtils = MakGAUtils(
        topology,
        trafficDesign[0],
        fgrs
    )

    def decodePopWrapper(pop: list[Individual], topology: Topology, sfcr: list[SFCRequest]) -> list[DecodedIndividual]:
        return gahaUtils.decodePop(pop, ignoreVNFInstances=True)

    def generateRandomIndividual(container: Type[Individual], topology: Topology, sfcr: list[SFCRequest]) -> Individual:
        return gahaUtils.generateRandomIndividual(container)

    hybridEvolution: HybridEvolution = HybridEvolution(
        "gaha",
        decodePopWrapper,
        generateRandomIndividual,
        tools.cxTwoPoint,
        gahaUtils.mutate,
        Individual,
        mutPb,
        cxPb,
        indPb,
        evaluateOnline=evaluateOnline
    )

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
