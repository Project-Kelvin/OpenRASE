"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
from typing import Callable, Tuple
from uuid import uuid4
from deap import tools
import tensorflow as tf
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.models.embedding import DecodedIndividual, LinkData
from algorithms.surrogacy.constants.surrogate import (
    SURROGACY_PATH,
    SURROGATE_PATH,
)
from algorithms.surrogacy.solvers.chain_composition import generateFGs
from algorithms.surrogacy.solvers.link_embedding import EmbedLinks
from algorithms.surrogacy.solvers.vnf_embedding import generateEGs
from algorithms.surrogacy.utils.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution, Individual
from algorithms.surrogacy.utils.hybrid_evaluation import HybridEvaluation
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()


def decodePop(
    pop: "list[list[float]]", topology: Topology, sfcrs: "list[SFCRequest]"
) -> list[DecodedIndividual]:
    """
    Generates the Embedding Graphs.

    Parameters:
        individual (list[float]): the individual.
        topology (Topology): the topology.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        list[DecodedIndividual]: A list consisting of tuples containing the embedding graphs, embedding data, link data, and acceptance ratio.
    """

    decodedPop: "list[DecodedIndividual]" = []

    for i, individual in enumerate(pop):
        weights: "Tuple[list[float], list[float], list[float], list[float]]" = (
            getWeights(individual, sfcrs, topology)
        )
        ccWeights: "list[float]" = weights[0]
        ccBias: "list[float]" = weights[1]
        vnfWeights: "list[float]" = weights[2]
        vnfBias: "list[float]" = weights[3]
        linkWeights: "list[float]" = weights[4]
        linkBias: "list[float]" = weights[5]
        fgs: "list[EmbeddingGraph]" = generateFGs(sfcrs, ccWeights, ccBias)
        copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
        egs, nodes, embedData = generateEGs(copiedFGs, topology, vnfWeights, vnfBias)
        embedLinks: EmbedLinks = None
        linkData: LinkData = None
        if len(egs) > 0:
            embedLinks = EmbedLinks(topology, egs, linkWeights, linkBias)
            egs = embedLinks.embedLinks(nodes)
            linkData = embedLinks.getLinkData()
        ar: float = len(egs) / len(sfcrs)
        decodedPop.append((i, egs, embedData, linkData, ar))

    return decodedPop

def generateRandomIndividual(
    container: Individual, topology: Topology, sfcrs: "list[SFCRequest]"
) -> Individual:
    """
    Generates a random individual.

    Parameters:
        container (Individual): the container for the individual.
        topology (Topology): the topology.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        Individual: An individual randomly generated.
    """

    individual = container()
    individual.id = uuid4()
    weightLength: int = getWeightLength(sfcrs, topology)
    for _ in range(weightLength):
        individual.append(random.uniform(-1, 1))
    return Individual(individual)

def crossover(
    ind1: Individual,
    ind2: Individual,
) -> Tuple[Individual, Individual]:
    """
    Crossover between two individuals.

    Parameters:
        ind1 (Individual): the first individual.
        ind2 (Individual): the second individual.

    Returns:
        tuple[Individual, Individual]: the two individuals after crossover.
    """

    return tools.cxBlend(ind1, ind2, alpha=0.5)

def mutate(
    individual: Individual,
    indpb: float,
) -> Individual:
    """
    Mutates an individual.

    Parameters:
        individual (Individual): the individual to mutate.
        indpb (float): the independent probability for each attribute to be mutated.

    Returns:
        Individual: the mutated individual.
    """

    return tools.mutGaussian(individual, mu=0.0, sigma=1.0, indpb=indpb)

hybridEvolution: HybridEvolution = HybridEvolution(
    "genesis",
    decodePop,
    generateRandomIndividual,
    crossover,
    mutate
)

def solve(
    sfcrs: "list[SFCRequest]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    experimentName: str
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Service Function Chains.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        experimentName (str): the name of the experiment.

    Returns:
        None
    """

    hybridEvolution.hybridSolve(
        topology,
        sfcrs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        experimentName
    )
