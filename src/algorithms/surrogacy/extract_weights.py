"""
This defines the functions used to extract weight values from individuals.
"""

from typing import Tuple
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.utils.config import getConfig

def getCCWeights(sfcrs: "list[SFCRequest]") -> int:
    """
    Gets the number of CC weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        int: the number of CC weights.
    """

    return len(sfcrs) + len(getConfig()["vnfs"]["names"])

def getLinkWeight(sfcrs: "list[SFCRequest]", topology: Topology) -> int:
    """
    Gets the number of link weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of link weights.
    """

    return len(sfcrs) + 2 * (len(topology["hosts"]) + len(topology["switches"]) + 2)

def getVNFWeight(sfcrs: "list[SFCRequest]", topology: Topology) -> int:
    """
    Gets the number of VNF weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of VNF weights.
    """

    return len(sfcrs) + len(getConfig()["vnfs"]["names"]) + len(topology["hosts"]) + 1

def getCCBias() -> int:
    """
    Gets the number of CC biases.

    Returns:
        int: the number of CC biases.
    """

    return 1

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

def getWeightLength(sfcrs: "list[SFCRequest]", topology: Topology) -> int:
    """
    Gets the number of weights.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of weights.
    """

    return (
        getCCWeights(sfcrs)
        + getLinkWeight(sfcrs, topology)
        + getVNFWeight(sfcrs, topology)
        + getCCBias()
        + getLinkBias()
        + getVNFBias()
    )

def getWeights(
    individual: "list[float]", sfcrs: "list[SFCRequest]", topology: Topology
) -> "Tuple[list[float], list[float], list[float], list[float]]":
    """
    Gets the weights.

    Parameters:
        individual (list[float]): the individual.
        sfcrs (list[SFCRequest]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: VNF weights, VNF bias, link weights, link bias.
    """

    ccWeights: int = getCCWeights(sfcrs)
    vnfWeights: int = getVNFWeight(sfcrs, topology)
    linkWeights: int = getLinkWeight(sfcrs, topology)
    ccBias: int = getCCBias()
    vnfBias: int = getVNFBias()
    linkBias: int = getLinkBias()

    ccWeightUpper: int = ccWeights
    ccBiasUpper: int = ccWeightUpper + ccBias
    vnfWeightUpper: int = ccBiasUpper + vnfWeights
    vnfBiasUpper: int = vnfWeightUpper + vnfBias
    linkWeightUpper: int = vnfBiasUpper + linkWeights
    linkBiasUpper: int = linkWeightUpper + linkBias

    return (
        individual[0:ccWeightUpper],
        individual[ccWeightUpper:ccBiasUpper],
        individual[ccBiasUpper:vnfWeightUpper],
        individual[vnfWeightUpper:vnfBiasUpper],
        individual[vnfBiasUpper:linkWeightUpper],
        individual[linkWeightUpper:linkBiasUpper],
    )
