"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from copy import deepcopy
import random
from typing import Callable, Type
from algorithms.utils.graphs import convertSFCRsToEGs
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
from algorithms.mak_ga.gaha_individual import generateRandomIndividual as generateRandomGAHAIndividual

POP_SIZE: int = 20
INDPB: float = 0.7 # Experimentally determined gene mutation probability for the GA
MUTPB: float = 0.7 # Experimentally determined mutation probability for the GA
CXPB: float = 1.0

def solve(
    topology: Topology,
    sfcrs: "list[SFCRequest]",
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
    retrain: bool = False,
    linesToWrite: list[str] = []
) -> None:
    """
    Solves the problem using a GA for VNF embedding and Dijkstra for link embedding.

    Parameters:
        topology (Topology): the topology to use for solving.
        sfcrs (list[SFCRequest]): the list of SFC requests to embed.
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
        retrain (bool): whether to retrain the surrogate model.
        linesToWrite (list[str]): list of lines to write to the log file.

    Returns:
        None
    """

    shuffledSFCRs: "list[SFCRequest]" = deepcopy(sfcrs)
    for sfcr in shuffledSFCRs:
        random.shuffle(sfcr["vnfs"])

    gahaUtils: MakGAUtils = MakGAUtils(
        topology,
        trafficDesign[0],
        shuffledSFCRs
    )

    def decodePopWrapper(pop: list[Individual], topology: Topology, sfcr: list[SFCRequest]) -> list[DecodedIndividual]:
        return gahaUtils.decodePop(pop, ignoreVNFInstances=True)

    def generateRandomIndividual(container: Type[Individual], topology: Topology, sfcr: list[SFCRequest]) -> Individual:
        return generateRandomGAHAIndividual(container, convertSFCRsToEGs(sfcr), topology)

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
        evaluateOnline=evaluateOnline,
        retrain=retrain,
        rejectVNF=gahaUtils.rejectVNF,
    )

    hybridEvolution.hybridSolve(
        topology,
        shuffledSFCRs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        telemetry,
        POP_SIZE,
        experiment,
        linesToWrite=linesToWrite
    )
