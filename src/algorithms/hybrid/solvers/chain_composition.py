"""
This defines the Chain Composition Algorithm.
"""

import numpy as np
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.utils.config import getConfig
from algorithms.hybrid.utils.solvers import activationFunction
from constants.topology import SERVER


def convertSFCRsToNP(sfcrs: "list[SFCRequest]") -> np.ndarray:
    """
    Converts a list of SFCRequests to a Numpy array.

    Parameters:
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        np.ndarray: the NumPy array.
    """

    data: "list[list[int]]" = []
    availableVNFs: "list[str]" = getConfig()["vnfs"]["names"]

    for sfcrIndex, sfcr in enumerate(sfcrs):
        sfcrHotCode: "list[int]" = [0] * len(sfcrs)
        sfcrHotCode[sfcrIndex] = 1

        for vnf in sfcr["vnfs"]:
            vnfsHotCode: "list[int]" = [0] * len(availableVNFs)
            vnfsHotCode[availableVNFs.index(vnf)] = 1

            data.append(
                sfcrHotCode
                + vnfsHotCode
            )

    return np.array(data, dtype=np.float64)

def getPriorityValue(data: np.ndarray, predefinedWeights: "list[float]", weights: list[float], noOfNeurons: int) -> np.array:
    """
    Get the priority value of VNFs using a NN.

    Parameters:
        data (np.ndarray): the data.
        predefinedWeights (list[float]): the predefined weights.
        weights (list[float]): the weights.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        np.ndarray: Array containing the priority values of VNFs.
    """

    copiedData = data.copy()
    npPDWeights: np.ndarray = np.array(predefinedWeights, dtype=np.float64)
    npPDWeights = npPDWeights.reshape(-1, noOfNeurons)
    copiedData = np.matmul(copiedData, npPDWeights)
    copiedData = activationFunction(copiedData)
    npWeights: np.ndarray = np.array(weights, dtype=np.float64).reshape(-1, 1)
    priorityValues: np.ndarray = np.matmul(copiedData, npWeights)
    priorityValues = activationFunction(priorityValues)

    return priorityValues

def sortVNFs(vnfs: "list[str]", order: "list[str]") -> "list[str]":
    """
    Sorts the VNFs according to the given order.

    Parameters:
        vnfs (list[str]): the VNFs.
        order (list[str]): the order.

    Returns:
        list[str]: the sorted VNFs.
    """

    lastVNFIndex: int = -1
    for vnf in order:
        vnfIndex: int = vnfs.index(vnf)
        if vnfIndex < lastVNFIndex:
            vnfs.remove(vnf)
            vnfIndex = lastVNFIndex
            vnfs.insert(vnfIndex, vnf)

        lastVNFIndex = vnfIndex

    return vnfs

def convertNPtoFGs(
    priorityValues: np.ndarray, sfcrs: "list[SFCRequest]"
) -> "dict[str, list[str]]":
    """
    Converts a Numpy array to a list of EmbeddingGraphs.

    Parameters:
        priorityValues (np.ndarray): the priority values.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        dict[str, list[str]]: the dictionary of ordered VNFs in every SFCR.
    """

    startIndex: int = 0
    fgs: dict[str, list[str]] = {}
    for sfcr in sfcrs:
        endIndex: int = startIndex + len(sfcr["vnfs"])
        vnfPriority: np.ndarray = priorityValues[startIndex:endIndex]
        priorityNP: np.ndarray = np.concatenate(
            (
                np.array(sfcr["vnfs"]).reshape(-1, 1),
                np.array(vnfPriority, dtype=np.float64).reshape(-1, 1)
            ),
            axis=1
        )
        priorityNP = priorityNP[priorityNP[:, 1].argsort()[::-1]]
        sortedVNFs: "list[str]" = priorityNP[:, 0].tolist()
        if "strictOrder" in sfcr and len(sfcr["strictOrder"]) > 0:
            sortedVNFs = sortVNFs(sortedVNFs, sfcr["strictOrder"])

        fgs[sfcr["sfcrID"]] = sortedVNFs

        startIndex = endIndex

    return fgs

def generateFGs(sfcrs: "list[SFCRequest]", pdWeights: "list[float]", weights: "list[float]", noOfNeurons: int) -> dict[str, list[str]]:
    """
    Generates the EmbeddingGraphs for the given SFCRequests.

    Parameters:
        sfcrs (list[SFCRequest]): the SFCRequests.
        pdWeights (list[float]): the predefined weights.
        weights (list[float]): the weights.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        dict[str, list[str]]: the dictionary of ordered VNFs in every SFCR.
    """

    # Convert the SFCRequests to a Numpy array
    data: np.ndarray = convertSFCRsToNP(sfcrs)

    # Get the priority value
    data = getPriorityValue(data, pdWeights, weights, noOfNeurons)

    # Convert the Numpy array to a list of EmbeddingGraphs
    forwardingGraphs: dict[str, list[str]] = convertNPtoFGs(data, sfcrs)

    return forwardingGraphs
