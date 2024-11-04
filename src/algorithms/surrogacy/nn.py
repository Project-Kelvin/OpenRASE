"""
This defines the Neural Network used for genetic encoding.
"""

import copy
from typing import Tuple, Union
import pandas as pd
from algorithms.surrogacy.local_constants import BRANCH
from constants.topology import SERVER, SFCC
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig

from utils.embedding_graph import traverseVNF
import tensorflow as tf
import numpy as np

def convertFGstoDF(fgs: "list[EmbeddingGraph]", topology: Topology) -> pd.DataFrame:
    """
    Converts a list of EmbeddingGraphs to a Pandas Dataframe.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of EmbeddingGraphs.
        topology (Topology): the topology.

    Returns:
        pd.DataFrame: the Pandas Dataframe.
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

        columns: "list[str]" = []

        for i in range(len(fgs)):
            columns.append(f"SFC{i}")
        for i in range(len(vnfs)):
            columns.append(f"VNF{i}")
        for i in range(len(topology["hosts"])):
            columns.append(f"Host{i}")
        columns.append("Position")

    return pd.DataFrame(data, columns=columns)


def convertDFtoFGs(data: pd.DataFrame, fgs: "list[EmbeddingGraph]", topology: Topology) -> "Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]]]":
    """
    Generates the Embedding Graphs.

    Parameters:
        data (pd.DataFrame): the data.
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
        fg["sfcID"] = fg["sfcrID"] if "sfcrID" in fg else f"sfc{index}"
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

            cls: "list[float]" = data[startIndex[0] : endIndex[0]][
                "ConfidenceLevel"
            ].tolist()
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

    return (egs, nodes, embeddingData)


def getConfidenceValues(data: pd.DataFrame, weights: "list[float]", bias: "list[float]") -> pd.DataFrame:
    """
    Gets the confidence values.

    Parameters:
        data (pd.DataFrame): the data.
        weights (list[list[float]]): the weights.
        bias (list[float]): the bias.

    Returns:
        pd.DataFrame: the confidence values.
    """

    layers: "list[int]" = [len(data.columns), 1]
    copiedData = data.copy()

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(layers[0], )),
        tf.keras.layers.Dense(layers[1], activation='sigmoid'),
    ])

    index: int = 0
    wStartIndex: int = 0
    wEndIndex: int = layers[index] * layers[index + 1]
    bStartIndex: int = 0
    bEndIndex: int = layers[index + 1]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([
                np.array(weights[wStartIndex:wEndIndex]).reshape(layers[index], layers[index + 1]),
                np.array(bias[bStartIndex:bEndIndex]).reshape(layers[index + 1])
            ])
            index += 1
            if index < len(layers) - 1:
                wStartIndex = wEndIndex
                wEndIndex = wStartIndex + (layers[index] * layers[index + 1])
                bStartIndex = bEndIndex
                bEndIndex = bStartIndex + layers[index + 1]

    prediction = model.predict(np.array(copiedData))
    copiedData = copiedData.assign(ConfidenceLevel=prediction)

    return copiedData
