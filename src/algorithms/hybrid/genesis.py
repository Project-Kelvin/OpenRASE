"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from concurrent.futures import ProcessPoolExecutor
import random
import timeit
from typing import Callable, Tuple
from uuid import uuid4
from deap import tools
import numpy as np
import tensorflow as tf
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.models.embedding import DecodedIndividual, LinkData
from algorithms.hybrid.constants.surrogate import (
    SURROGACY_PATH,
    SURROGATE_PATH,
)
from algorithms.hybrid.solvers.chain_composition import generateFGs
from algorithms.hybrid.solvers.link_embedding import EmbedLinks
from algorithms.hybrid.solvers.vnf_embedding import generateEGs
from algorithms.hybrid.utils.extract_weights import (
    generatePredefinedWeights,
    generateRandomWeight,
    getWeights,
    getPredefinedWeights,
    getWeightsLength,
)
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution, Individual
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

predefinedWeights: tuple[list[float], list[float], list[float]] = None
NO_OF_NEURONS: int = 2
POP_SIZE: int = 100


def decodeIndividual(
    individual: "list[float]",
    index: int,
    topology: Topology,
    sfcrs: "list[SFCRequest]",
) -> DecodedIndividual:
    """
    Decodes an individual to an Embedding Graph.

    Parameters:
        individual (list[float]): the individual.
        index (int): the index of the individual.
        topology (Topology): the topology.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        DecodedIndividual: A tuple containing the embedding graphs, embedding data, link data, and acceptance ratio.
    """

    global predefinedWeights

    # predefinedIndividual = individual[:-getWeightsLength(NO_OF_NEURONS)] if NO_OF_NEURONS > 0 else individual
    # predefinedWeights: "tuple[list[float], list[float], list[float]]" = getPredefinedWeights(
    #     predefinedIndividual, sfcrs, topology, NO_OF_NEURONS
    # )

    weights: "Tuple[list[float], list[float], list[float]]" = getWeights(
        individual, NO_OF_NEURONS
    )

    ccPDWeights: "list[float]" = predefinedWeights[0]
    vnfPDWeights: "list[float]" = predefinedWeights[1]
    linkPDWeights: "list[float]" = predefinedWeights[2]
    ccWeights: list[float] = weights[0]
    vnfWeights: list[float] = weights[1]
    linkWeights: list[float] = weights[2]

    fgs: dict[str, list[str]] = generateFGs(
        sfcrs, ccPDWeights, ccWeights, NO_OF_NEURONS
    )
    egs, nodes, embedData = generateEGs(
        fgs, topology, vnfPDWeights, vnfWeights, NO_OF_NEURONS
    )
    embedLinks: EmbedLinks = None
    linkData: LinkData = None
    if len(egs) > 0:
        embedLinks = EmbedLinks(
            topology, sfcrs, egs, linkPDWeights, linkWeights, NO_OF_NEURONS
        )
        egs = embedLinks.embedLinks(nodes)
        linkData = embedLinks.getLinkData()
    ar: float = len(egs) / len(sfcrs)

    return (index, egs, embedData, linkData, ar)


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

    startTime: int = timeit.default_timer()
    decodedPop: "list[DecodedIndividual]" = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(decodeIndividual, individual, index, topology, sfcrs)
            for index, individual in enumerate(pop)
        ]

        for future in futures:
            decodedPop.append(future.result())

    endTime: int = timeit.default_timer()
    TUI.appendToSolverLog(
        f"Decoded {len(decodedPop)} individuals in {endTime - startTime:.2f} seconds."
    )

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

    # individual.extend(generatePredefinedWeights(
    #     sfcrs, topology, NO_OF_NEURONS
    # ))

    weightLength: int = getWeightsLength(NO_OF_NEURONS)
    for _ in range(weightLength):
        individual.append(generateRandomWeight())

    return individual

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

    return tools.mutGaussian(individual, mu=0.0, sigma=np.pi, indpb=indpb)


def solve(
    sfcrs: "list[SFCRequest]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: list[TrafficDesign],
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    dirName: str,
    experimentName: str,
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Service Function Chains.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (list[TrafficDesign]): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        dirName (str): the directory name.
        experimentName (str): the name of the experiment.

    Returns:
        None
    """

    global predefinedWeights
    predefinedWeights = getPredefinedWeights(
        generatePredefinedWeights(sfcrs, topology, NO_OF_NEURONS),
        sfcrs,
        topology,
        NO_OF_NEURONS,
    )

    hybridEvolution: HybridEvolution = HybridEvolution(
        dirName, decodePop, generateRandomIndividual, crossover, mutate
    )

    hybridEvolution.hybridSolve(
        topology,
        sfcrs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        POP_SIZE,
        experimentName,
    )
