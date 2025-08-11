"""
This defines the functions used to extract weight values from individuals.
"""

import random
from typing import Tuple
import numpy as np
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.utils.config import getConfig

from algorithms.hybrid.solvers.link_embedding import EmbedLinks

def getCCWeightsLength(sfcrs: "list[SFCRequest]") -> int:
    """
    Gets the number of CC weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        int: the number of CC weights.
    """

    return len(sfcrs) + len(getConfig()["vnfs"]["names"])

def getLinkWeightsLength(sfcrs: "list[SFCRequest]", topology: Topology) -> int:
    """
    Gets the number of link weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of link weights.
    """

    return len(sfcrs) + len(EmbedLinks.getLinks(topology))

def getVNFWeightsLength(sfcrs: "list[SFCRequest]", topology: Topology) -> int:
    """
    Gets the number of VNF weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of VNF weights.
    """

    return len(sfcrs) + len(getConfig()["vnfs"]["names"]) + 1

def getPredefinedWeightsLength(sfcrs: "list[SFCRequest]", topology: Topology, noOfNeurons: int) -> int:
    """
    Gets the number of weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        int: the number of weights.
    """

    if noOfNeurons == 0:
        noOfNeurons = 1

    return noOfNeurons * (
        getCCWeightsLength(sfcrs)
        + getLinkWeightsLength(sfcrs, topology)
        + getVNFWeightsLength(sfcrs, topology)
    )

def getWeightsLength(noOfNeurons: int) -> int:
    """
    Gets the total number of weights.

    Parameters:
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        int: the total number of weights.
    """

    return 3 * noOfNeurons

def generatePredefinedWeights(sfcrs: "list[SFCRequest]", topology: Topology, noOfNeurons: int) -> "list[float]":
    """
    Generates the predefined weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of SFCRequests.
        topology (Topology): the topology.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        list[float]: the predefined weights.
    """

    totalWeights: int = getPredefinedWeightsLength(sfcrs, topology, noOfNeurons)
    return [generateRandomWeight() for _ in range(totalWeights)]


def getPredefinedWeights(
    predefinedWeights: list[float],
    sfcrs: "list[SFCRequest]",
    topology: Topology,
    noOfNeurons: int,
) -> "Tuple[list[float], list[float], list[float]]":
    """
    Gets the weights.

    Parameters:
        predefinedWeights (list[float]): the predefined weights.
        sfcrs (list[SFCRequest]): the list of SFCRequests.
        topology (Topology): the topology.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        tuple[list[float], list[float], list[float]]: VNF weights, VNF bias, link weights.
    """

    if noOfNeurons == 0:
        noOfNeurons = 1

    ccWeights: int = getCCWeightsLength(sfcrs) * noOfNeurons
    vnfWeights: int = getVNFWeightsLength(sfcrs, topology) * noOfNeurons
    linkWeights: int = getLinkWeightsLength(sfcrs, topology) * noOfNeurons

    ccWeightUpper: int = ccWeights
    vnfWeightUpper: int = ccWeightUpper + vnfWeights
    linkWeightUpper: int = vnfWeightUpper + linkWeights

    return (
        predefinedWeights[0:ccWeightUpper],
        predefinedWeights[ccWeightUpper:vnfWeightUpper],
        predefinedWeights[vnfWeightUpper:linkWeightUpper],
    )


def getWeights(
    individual: "list[float]", noOfNeurons: int
) -> "Tuple[list[float], list[float], list[float]]":
    """
    Gets the weights.

    Parameters:
        individual (list[float]): the individual.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        tuple[list[float], list[float], list[float]]: CC weights, VNF weights, link weights.
    """

    ccWeightUpper: int = noOfNeurons
    vnfWeightUpper: int = ccWeightUpper + noOfNeurons
    linkWeightUpper: int = vnfWeightUpper + noOfNeurons
    return (
        individual[0:ccWeightUpper],
        individual[ccWeightUpper:vnfWeightUpper],
        individual[vnfWeightUpper:linkWeightUpper],
    )

def generateRandomWeight() -> float:
    """
    Generates a random weight.

    Returns:
        float: a random weight.
    """

    return random.uniform(-1 * np.pi, 1 * np.pi)
