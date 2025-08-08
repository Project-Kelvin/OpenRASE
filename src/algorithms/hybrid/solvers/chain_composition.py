"""
This defines the Chain Composition Algorithm.
"""

import numpy as np
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.utils.config import getConfig
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

def getPriorityValue(data: np.ndarray, weights: "list[float]", bias: int) -> np.array:
    """
    Get the priority value of VNFs using a NN.

    Parameters:
        data (np.ndarray): the data.
        weights (list[float]): the weights.
        bias (int): the bias.

    Returns:
        np.ndarray: Array containing the priority values of VNFs.
    """

    copiedData = data.copy()
    npWeights = np.array(weights, dtype=np.float64)
    npWeights = npWeights.reshape(-1, 1)
    copiedData = np.matmul(copiedData, npWeights)
    priorityValue = copiedData[:, 0] + bias

    return priorityValue

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

def generateEG(sortedVNFs: "list[str]", sfcrID: str) -> EmbeddingGraph:
    """
    Generates the EmbeddingGraph for the given SFCRequest.

    Parameters:
        sortedVNFs (list[str]): the sorted VNFs.
        sfcrID (str): the SFCRequest ID.

    Returns:
        None
    """

    forwardingGraph: EmbeddingGraph = {
            "sfcID": sfcrID,
            "vnfs": {}
        }

    vnfDict: VNF = forwardingGraph["vnfs"]
    splitters: "list[str]" = getConfig()["vnfs"]["splitters"]

    def addVNF(vnfs: "list[str]", vnfDict: VNF, splitters: "list[str]") -> None:
        """
        Adds VNF to the EmbeddingGraph.

        Parameters:
            vnfs (list[str]): the VNFs.
            vnfDict (VNF): the VNF dictionary.
            splitters (list[str]): the list of splitters.

        Returns:
            None
        """

        if len(vnfs) == 0:
            vnfDict["host"] = {
                "id": SERVER
            }
            vnfDict["next"] = TERMINAL

            return

        vnf: str = vnfs.pop(0)
        splitter: bool = vnf in splitters
        vnfDict["vnf"] = {
            "id": vnf
        }
        vnfDict["next"] = [{}, {}] if splitter else {}

        if splitter:
            for i in range(2):

                addVNF(vnfs.copy(), vnfDict["next"][i], splitters)
        else:
            addVNF(vnfs, vnfDict["next"], splitters)

    addVNF(sortedVNFs, vnfDict, splitters)

    return forwardingGraph

def convertNPtoFGs(
    priorityValues: np.ndarray, sfcrs: "list[SFCRequest]"
) -> "list[EmbeddingGraph]":
    """
    Converts a Numpy array to a list of EmbeddingGraphs.

    Parameters:
        priorityValues (np.ndarray): the priority values.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        list[EmbeddingGraph]: the list of EmbeddingGraphs.
    """

    forwardingGraphs: "list[EmbeddingGraph]" = []
    startIndex: int = 0
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

        forwardingGraphs.append(generateEG(sortedVNFs, sfcr["sfcrID"]))
        startIndex = endIndex

    return forwardingGraphs

def generateFGs(sfcrs: "list[SFCRequest]", weights: "list[float]", bias: "list[float]") -> list[EmbeddingGraph]:
    """
    Generates the EmbeddingGraphs for the given SFCRequests.

    Parameters:
        sfcrs (list[SFCRequest]): the SFCRequests.
        weights (list[float]): the weights.
        bias (list[float]): the bias.

    Returns:
        list[EmbeddingGraph]: the list of EmbeddingGraphs.
    """

    # Convert the SFCRequests to a Numpy array
    data: np.ndarray = convertSFCRsToNP(sfcrs)

    # Get the priority value
    data = getPriorityValue(data, weights, bias)

    # Convert the Numpy array to a list of EmbeddingGraphs
    forwardingGraphs: "list[EmbeddingGraph]" = convertNPtoFGs(data, sfcrs)

    return forwardingGraphs
