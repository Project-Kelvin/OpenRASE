"""
This defines the Neural Network used for genetic encoding.
"""

import copy
import random
from typing import Tuple
from shared.constants.embedding_graph import TERMINAL
from algorithms.hybrid.constants.surrogate import BRANCH
from algorithms.hybrid.utils.solvers import activationFunction
from constants.topology import SERVER, SFCC
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig

from utils.embedding_graph import traverseVNF
import numpy as np

def convertFGsToNP(fgs: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """
    Converts a list of EmbeddingGraphs to a NumPy array.

    Parameters:
        fgs (dict[str, list[str]]): the dictionary of EmbeddingGraphs.

    Returns:
        tuple[np.ndarray, list[str]]: (the NumPy array, the list of indices).
    """

    vnfs: "list[str]" = getConfig()["vnfs"]["names"]
    data: "list[list[int]]" = []
    totalInstances: int = 2
    indices: list[str] = []

    for index, fg in enumerate(fgs.items()):
        fgID: int = index
        fgHotCode: "list[int]" = [0] * len(fgs)
        fgHotCode[fgID] = 1

        for vnf in fg[1]:
            vnfID: int = vnfs.index(vnf)
            vnfsHotCode: "list[int]" = [0] * len(vnfs)
            vnfsHotCode[vnfID] = 1

            for instances in range(0, totalInstances):
                instanceHotCode: "list[int]" = [0] * totalInstances
                instanceHotCode[instances] = 1
                row: "list[int]" = fgHotCode + vnfsHotCode + instanceHotCode
                data.append(row)
                indices.append(f"{fg[0]}_{vnf}_{instances}")

    return np.array(data, dtype=np.float64), indices


def convertNPtoEGs(
    data: np.ndarray, fgs: dict[str, list[str]], topology: Topology, indices: list[str]
) -> "Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]":
    """
    Generates the Embedding Graphs.

    Parameters:
        data (np.ndarray): the data.
        fgs (dict[str, list[str]]): the dictionary of Embedding Graphs.
        topology (Topology): the topology.
        indices (list[str]): the list of indices.
    Returns:
        Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]: (the Embedding Graphs, hosts in the order they should be linked, the embedding data containing the VNFs in hosts).
    """

    noHosts: int = len(topology["hosts"])
    egs: "list[EmbeddingGraph]" = []
    nodes: "dict[str, list[str]]" = {}
    embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]" = {}
    splitters: "list[str]" = getConfig()["vnfs"]["splitters"]
    for sfcrID, sortedVNFs in fgs.items():
        forwardingGraph: EmbeddingGraph = {"sfcID": sfcrID, "vnfs": {}}
        nodes[sfcrID] = [SFCC]
        embeddingNotFound: bool = False
        oldDepth: int = 1
        depth: int = 1
        vnfDict: VNF = forwardingGraph["vnfs"]

        def addVNF(vnfs: "list[str]", vnfDict: VNF, depth: int, instance: int) -> None:
            """
            Adds VNF to the EmbeddingGraph.

            Parameters:
                vnfs (list[str]): the VNFs.
                vnfDict (VNF): the VNF dictionary.
                depth (int): the depth of the VNF.
                instance (int): the instance number.

            Returns:
                None
            """

            nonlocal oldDepth, indices, nodes, embeddingData, embeddingNotFound, splitters

            if depth != oldDepth:
                oldDepth = depth
                if nodes[sfcrID][-1] != SERVER:
                    # pylint: disable=cell-var-from-loop
                    nodes[sfcrID].append(BRANCH)

            if embeddingNotFound:
                return

            if len(vnfs) == 0:
                vnfDict["host"] = {"id": SERVER}
                vnfDict["next"] = TERMINAL
                nodes[sfcrID].append(SERVER)

                return

            vnf: str = vnfs.pop(0)
            splitter: bool = vnf in splitters
            vnfDict["next"] = [{}, {}] if splitter else {}
            index: int = indices.index(f"{sfcrID}_{vnf}_{instance}")
            cl: float = data[index, 0]

            # Reject 5% of the VNFs
            rejectionRate: float = 0.05
            absCL: float = abs(cl)
            acceptanceValue: float = absCL % int(absCL) if int(absCL) != 0 else absCL

            if acceptanceValue < rejectionRate:
                embeddingNotFound = True

                return
            else:
                hostIndex = abs(int(random.gauss(cl, 2)))
                hostIndex = hostIndex % noHosts

                vnfDict["vnf"] = {"id": vnf}
                vnfDict["host"] = {"id": topology["hosts"][hostIndex]["id"]}

                # pylint: disable=cell-var-from-loop
                if nodes[sfcrID][-1] != vnfDict["host"]["id"]:
                    # pylint: disable=cell-var-from-loop
                    nodes[sfcrID].append(vnfDict["host"]["id"])

                if vnfDict["host"]["id"] in embeddingData:
                    if sfcrID in embeddingData[vnfDict["host"]["id"]]:
                        embeddingData[vnfDict["host"]["id"]][sfcrID].append(
                            [vnfDict["vnf"]["id"], depth]
                        )
                    else:
                        embeddingData[vnfDict["host"]["id"]][sfcrID] = [
                            [vnfDict["vnf"]["id"], depth]
                        ]
                else:
                    embeddingData[vnfDict["host"]["id"]] = {
                        sfcrID: [[vnfDict["vnf"]["id"], depth]]
                    }

            if splitter:
                depth += 1
                for i in range(2):
                    addVNF(vnfs.copy(), vnfDict["next"][i], depth, i)
            else:
                addVNF(vnfs, vnfDict["next"], depth, 0)

        addVNF(sortedVNFs, vnfDict, depth, 0)
        if not embeddingNotFound:
            eg: EmbeddingGraph = copy.deepcopy(forwardingGraph)
            egs.append(eg)
        else:
            for hosts in embeddingData.values():
                if forwardingGraph["sfcID"] in hosts:
                    del hosts[forwardingGraph["sfcID"]]

    return (egs, nodes, embeddingData)


def getConfidenceValues(data: np.ndarray, predefinedWeights: "list[float]", weights: "list[float]", noOfNeurons: int, topology: Topology) -> np.ndarray:
    """
    Gets the confidence values.

    Parameters:
        data (np.ndarray): the data.
        predefinedWeights (list[float]): the predefined weights.
        weights (list[float]): the weights.
        noOfNeurons (int): the number of neurons in the hidden layer.
        topology (Topology): the topology.

    Returns:
        np.ndarray: the confidence values.
    """

    noOfHosts: int = len(topology["hosts"])
    copiedData = data.copy()
    npWeights = np.array(predefinedWeights, dtype=np.float64).reshape(-1, noOfNeurons if noOfNeurons > 0 else 1)
    confidenceValues: np.ndarray = np.matmul(copiedData, npWeights)
    confidenceValues = activationFunction(confidenceValues)
    if noOfNeurons > 0:
        npWeights = np.array(weights, dtype=np.float64).reshape(-1, 1)
        confidenceValues = np.matmul(confidenceValues, npWeights)
        confidenceValues = noOfHosts * activationFunction(confidenceValues)

    return confidenceValues

def generateEGs(
    fgs: dict[str, list[str]], topology: Topology, pdWeights: "list[float]", weights: "list[float]", noOfNeurons: int
) -> Tuple["list[EmbeddingGraph]", dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]:
    """
    Generates the Embedding Graphs.

    Parameters:
        fgs (dict[str, list[str]]): the dict of Embedding Graphs.
        topology (Topology): the topology.
        pdWeights (list[float]): the predefined weights.
        weights (list[float]): the weights.
        noOfNeurons (int): the number of neurons in the hidden layer.

    Returns:
        Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]: (the Embedding Graphs, hosts in the order they should be linked, the embedding data containing the VNFs in hosts).
    """

    data, indices = convertFGsToNP(fgs)
    data = getConfidenceValues(data, pdWeights, weights, noOfNeurons, topology)
    egs, nodes, embeddingData = convertNPtoEGs(data, fgs, topology, indices)

    return egs, nodes, embeddingData
