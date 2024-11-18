"""
This defines the functions used to extract weight values from individuals.
"""

from typing import Tuple
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig


def getLinkWeight(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of link weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of link weights.
    """

    return len(fgs) + 2 * (len(topology["hosts"]) + len(topology["switches"]) + 2)

def getVNFWeight(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of VNF weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of VNF weights.
    """

    return len(fgs) + len(getConfig()["vnfs"]["names"]) + len(topology["hosts"]) + 1

def getLinkBias() -> int:
    """
    Gets the number of link biases.

    Returns:
        int: the number of link biases.
    """

    return 1

def getVNFBias() -> int:
    """
    Gets the number of VNF biases.

    Returns:
        int: the number of VNF biases.
    """

    return 1

def getWeightLength(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of weights.
    """

    return (
        getLinkWeight(fgs, topology)
        + getVNFWeight(fgs, topology)
        + getLinkBias()
        + getVNFBias()
    )

def getWeights(
    individual: "list[float]", fgs: "list[EmbeddingGraph]", topology: Topology
) -> "Tuple[list[float], list[float], list[float], list[float]]":
    """
    Gets the weights.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: VNF weights, VNF bias, link weights, link bias.
    """

    vnfWeights: int = getVNFWeight(fgs, topology)
    linkWeights: int = getLinkWeight(fgs, topology)
    vnfBias: int = getVNFBias()
    linkBias: int = getLinkBias()

    vnfWeightUpper: int = vnfWeights
    vnfBiasUpper: int = vnfWeights + vnfBias
    linkWeightUpper: int = vnfWeights + vnfBias + linkWeights
    linkBiasUpper: int = vnfWeights + vnfBias + linkWeights + linkBias

    return (
        individual[0:vnfWeightUpper],
        individual[vnfWeightUpper:vnfBiasUpper],
        individual[vnfBiasUpper:linkWeightUpper],
        individual[linkWeightUpper:linkBiasUpper],
    )
