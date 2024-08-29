"""
This defines the Neural Network used for genetic encoding.
"""

import pandas as pd
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
    for index, fg in enumerate(fgs):
        fgID: int = index + 1

        def parseVNF(vnf: VNF, pos: int) -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                pos (int): the position.
            """

            vnfID: int = vnfs.index(vnf["vnf"]["id"]) + 1

            for hIndex, _host in enumerate(topology["hosts"]):
                #pylint: disable=cell-var-from-loop
                data.append([fgID, vnfID, pos, hIndex + 1])

        traverseVNF(fg["vnfs"], parseVNF, shouldParseTerminal=False)

    return pd.DataFrame(data, columns=["SFC", "VNF", "Position", "Host"])


def convertDFtoFGs(data: pd.DataFrame, fgs: "list[EmbeddingGraph]", topology: Topology) -> "list[EmbeddingGraph]":
    """
    Generates the Embedding Graphs.

    Parameters:
        data (pd.DataFrame): the data.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        list[EmbeddingGraph]: the Embedding Graphs.
    """

    noHosts: int = len(topology["hosts"])
    startIndex: "list[int]" = [0]
    endIndex: "list[int]" = [noHosts]

    egs: "list[EmbeddingGraph]" = []
    for _index, fg in enumerate(fgs):
        embeddingNotFound: "list[bool]" = [False]

        def parseVNF(vnf: VNF, _pos: int, embeddingNotFound, startIndex, endIndex) -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                _pos (int): the position.
                startIndex (list[int]): the start index.
                endIndex (list[int]): the end index.
            """

            if embeddingNotFound[0]:
                return

            cls: "list[float]" = data[startIndex[0]:endIndex[0]]["ConfidenceLevel"].tolist()
            startIndex[0] = startIndex[0] + noHosts
            endIndex[0] = endIndex[0] + noHosts

            maxCL: float = max(cls)

            if maxCL < 0.5:
                embeddingNotFound[0] = True

                return
            else:
                vnf["host"] = {
                    "id": topology["hosts"][cls.index(maxCL)]["id"]
                }

        traverseVNF(fg["vnfs"], parseVNF, embeddingNotFound, startIndex, endIndex, shouldParseTerminal=False)

        if not embeddingNotFound[0]:
            egs.append(fg)

    return egs


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

    layers: "list[int]" = [4, 1]
    copiedData = data.copy()
    normalizer = tf.keras.layers.Normalization(axis=1)
    normalizer.adapt(np.array(copiedData))

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(layers[0], )),
        normalizer,
        tf.keras.layers.Dense(layers[1], activation='sigmoid')
    ])

    index: int = 0
    startIndex: int = 0
    endIndex: int = layers[index]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([
                np.array(weights[startIndex:endIndex]).reshape(layers[index], layers[index + 1]),
                np.array([bias[index]])
            ])
            index += 1
            if index < len(layers) - 1:
                startIndex = endIndex
                endIndex = startIndex + layers[index]

    prediction = model.predict(np.array(copiedData))
    copiedData = copiedData.assign(ConfidenceLevel = prediction)

    return copiedData
