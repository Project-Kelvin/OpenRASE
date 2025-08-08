"""
This defines the Neural Network used for genetic encoding.
"""

import copy
from typing import Tuple
from algorithms.hybrid.constants.surrogate import BRANCH
from constants.topology import SERVER, SFCC
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig

from utils.embedding_graph import traverseVNF
import numpy as np

def convertFGsToNP(fgs: "list[EmbeddingGraph]", topology: Topology) -> np.ndarray:
    """
    Converts a list of EmbeddingGraphs to a NumPy array.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of EmbeddingGraphs.
        topology (Topology): the topology.

    Returns:
        np.ndarray: the NumPy array.
    """

    vnfs: "list[str]" = getConfig()["vnfs"]["names"]
    data: "list[list[int]]" = []
    instances: "dict[str, int]" = {}
    for index, fg in enumerate(fgs):
        fgID: int = index
        fgHotCode: "list[int]" = [0] * len(fgs)
        fgHotCode[fgID] = 1

        def parseVNF(vnf: VNF, _pos: int, instances: "dict[str, int]") -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                _pos (int): the position.
                instances (dict[str, int]): the instances.
            """

            vnfID: int = vnfs.index(vnf["vnf"]["id"])
            vnfsHotCode: "list[int]" = [0] * len(vnfs)
            vnfsHotCode[vnfID] = 1

            if vnf["vnf"]["id"] not in instances:
                instances[vnf["vnf"]["id"]] = 1
            else:
                instances[vnf["vnf"]["id"]] += 1

            for hIndex, _host in enumerate(topology["hosts"]):
                # pylint: disable=cell-var-from-loop
                hostHotCode: "list[int]" = [0] * len(topology["hosts"])
                hostHotCode[hIndex] = 1
                row: "list[int]" = fgHotCode + vnfsHotCode + hostHotCode + [instances[vnf["vnf"]["id"]]]
                data.append(row)

        traverseVNF(fg["vnfs"], parseVNF, instances, shouldParseTerminal=False)

    return np.array(data, dtype=np.float64)

def convertNPtoEGs(data: np.ndarray, fgs: "list[EmbeddingGraph]", topology: Topology) -> "Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]":
    """
    Generates the Embedding Graphs.

    Parameters:
        data (np.ndarray): the data.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]: (the Embedding Graphs, hosts in the order they should be linked, the embedding data containing the VNFs in hosts).
    """

    noHosts: int = len(topology["hosts"])
    startIndex: "list[int]" = [0]
    endIndex: "list[int]" = [noHosts]

    egs: "list[EmbeddingGraph]" = []
    nodes: "dict[str, list[str]]" = {}
    embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]" = {}

    for index, fg in enumerate(fgs):
        nodes[fg["sfcID"]] = [SFCC]
        embeddingNotFound: "list[bool]" = [False]
        oldDepth: int = 1

        def parseVNF(
            vnf: VNF, depth: int, embeddingNotFound, startIndex, endIndex) -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                depth (int): the depth.
                startIndex (list[int]): the start index.
                endIndex (list[int]): the end index.
            """

            nonlocal oldDepth

            if depth != oldDepth:
                oldDepth = depth
                if nodes[fg["sfcID"]][-1] != SERVER:
                    # pylint: disable=cell-var-from-loop
                    nodes[fg["sfcID"]].append(BRANCH)

            if embeddingNotFound[0]:
                return

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                # pylint: disable=cell-var-from-loop
                nodes[fg["sfcID"]].append(SERVER)

                return

            cls: "list[float]" = data[startIndex[0] : endIndex[0]].tolist()
            startIndex[0] = startIndex[0] + noHosts
            endIndex[0] = endIndex[0] + noHosts

            maxCL: float = max(cls)

            if maxCL < 0.1:
                embeddingNotFound[0] = True

                return
            else:
                vnf["host"] = {"id": topology["hosts"][cls.index(maxCL)]["id"]}

                # pylint: disable=cell-var-from-loop
                if nodes[fg["sfcID"]][-1] != vnf["host"]["id"]:
                    # pylint: disable=cell-var-from-loop
                    nodes[fg["sfcID"]].append(vnf["host"]["id"])

                if vnf["host"]["id"] in embeddingData:
                    if fg["sfcID"] in embeddingData[vnf["host"]["id"]]:
                        embeddingData[vnf["host"]["id"]][fg["sfcID"]].append([vnf["vnf"]["id"], depth])
                    else:
                        embeddingData[vnf["host"]["id"]][fg["sfcID"]] = [[vnf["vnf"]["id"], depth]]
                else:
                    embeddingData[vnf["host"]["id"]] = {
                        fg["sfcID"]: [[vnf["vnf"]["id"], depth]]
                    }

        traverseVNF(fg["vnfs"], parseVNF, embeddingNotFound, startIndex, endIndex)

        if not embeddingNotFound[0]:
            if "sfcrID" in fg:
                del fg["sfcrID"]

            eg: EmbeddingGraph = copy.deepcopy(fg)

            egs.append(eg)
        else:
            for hosts in embeddingData.values():
                if fg["sfcID"] in hosts:
                    del hosts[fg["sfcID"]]

    return (egs, nodes, embeddingData)


def getConfidenceValues(data: np.ndarray, weights: "list[float]", bias: "list[float]") -> np.ndarray:
    """
    Gets the confidence values.

    Parameters:
        data (np.ndarray): the data.
        weights (list[list[float]]): the weights.
        bias (list[float]): the bias.

    Returns:
        np.ndarray: the confidence values.
    """

    copiedData = data.copy()
    npWeights = np.array(weights, dtype=np.float64).reshape(-1, 1)
    confidenceValues: np.ndarray = np.matmul(copiedData, npWeights)
    confidenceValues = confidenceValues[:, 0] + bias

    return confidenceValues

def generateEGs(
    fgs: "list[EmbeddingGraph]", topology: Topology, weights: "list[float]", bias: "list[float]"
) -> Tuple["list[EmbeddingGraph]", dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]:
    """
    Generates the Embedding Graphs.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.
        weights (list[float]): the weights.
        bias (list[float]): the bias.

    Returns:
        Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]: (the Embedding Graphs, hosts in the order they should be linked, the embedding data containing the VNFs in hosts).
    """

    data: np.ndarray = convertFGsToNP(fgs, topology)
    data = getConfidenceValues(data, weights, bias)
    egs, nodes, embeddingData = convertNPtoEGs(data, fgs, topology)

    return egs, nodes, embeddingData
