"""
This defines the Chain Composition Algorithm.
"""

import numpy as np
import pandas as pd
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.utils.config import getConfig
import tensorflow as tf

from constants.topology import SERVER


def convertSFCRsToDF(sfcrs: "list[SFCRequest]") -> pd.DataFrame:
    """
    Converts a list of SFCRequests to a Pandas Dataframe.

    Parameters:
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        pd.DataFrame: the Pandas Dataframe.
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

    columns: "list[str]" = []

    for i in range(len(sfcrs)):
        columns.append(f"SFCR{i}")
    for i in range(len(availableVNFs)):
        columns.append(f"VNF{i}")

    return pd.DataFrame(data, columns=columns)

def getPriorityValue(data: pd.DataFrame, weights: "list[float]", bias: "list[float]") -> pd.DataFrame:
    """
    Get the priority value of VNFs using a NN.

    Parameters:
        data (pd.DataFrame): the data.
        weights (list[float]): the weights.
        bias (list[float]): the bias.

    Returns:
        pd.DataFrame: Data frame containing the priority values of VNFs.
    """

    layers: "list[int]" = [len(data.columns), 1]
    copiedData = data.copy()

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(layers[0],)),
            tf.keras.layers.Dense(layers[1], activation="relu"),
        ]
    )

    index: int = 0
    wStartIndex: int = 0
    wEndIndex: int = layers[index] * layers[index + 1]
    bStartIndex: int = 0
    bEndIndex: int = layers[index + 1]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights(
                [
                    np.array(weights[wStartIndex:wEndIndex]).reshape(
                        layers[index], layers[index + 1]
                    ),
                    np.array(bias[bStartIndex:bEndIndex]).reshape(layers[index + 1]),
                ]
            )
            index += 1
            if index < len(layers) - 1:
                wStartIndex = wEndIndex
                wEndIndex = wStartIndex + (layers[index] * layers[index + 1])
                bStartIndex = bEndIndex
                bEndIndex = bStartIndex + layers[index + 1]

    priority = model.predict(np.array(copiedData), verbose=0)
    copiedData = copiedData.assign(PriorityValue=priority)

    return copiedData

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

def convertDFtoFGs(
    df: pd.DataFrame, sfcrs: "list[SFCRequest]"
) -> "list[EmbeddingGraph]":
    """
    Converts a Pandas Dataframe to a list of EmbeddingGraphs.

    Parameters:
        df (pd.DataFrame): the Pandas Dataframe.
        sfcrs (list[SFCRequest]): the list of SFCRequests.

    Returns:
        list[EmbeddingGraph]: the list of EmbeddingGraphs.
    """

    forwardingGraphs: "list[EmbeddingGraph]" = []
    startIndex: int = 0
    for sfcr in sfcrs:
        endIndex: int = startIndex + len(sfcr["vnfs"])
        vnfPriority: "list[float]" = df[startIndex:endIndex]["PriorityValue"].to_list()
        priorityDF: pd.DataFrame = pd.DataFrame(
            {
                "vnf": sfcr["vnfs"],
                "priority": vnfPriority,
            }
        )
        priorityDF = priorityDF.sort_values(by="priority", ascending=False)
        sortedVNFs: "list[str]" = priorityDF["vnf"].to_list()

        if "strictOrder" in sfcr and len(sfcr["strictOrder"]) > 0:
            sortedVNFs = sortVNFs(sortedVNFs, sfcr["strictOrder"])

        forwardingGraphs.append(generateEG(sortedVNFs, sfcr["sfcrID"]))
        startIndex = endIndex

    return forwardingGraphs

def generateFGs(sfcrs: "list[SFCRequest]", weights: "list[float]", bias: "list[float]") -> pd.DataFrame:
    """
    Generates the EmbeddingGraphs for the given SFCRequests.

    Parameters:
        sfcrs (list[SFCRequest]): the SFCRequests.
        weights (list[float]): the weights.
        bias (list[float]): the bias.

    Returns:
        pd.DataFrame: the Pandas Dataframe.
    """

    # Convert the SFCRequests to a Pandas Dataframe
    data: pd.DataFrame = convertSFCRsToDF(sfcrs)

    # Get the priority value
    data = getPriorityValue(data, weights, bias)

    # Convert the Pandas Dataframe to a list of EmbeddingGraphs
    forwardingGraphs: "list[EmbeddingGraph]" = convertDFtoFGs(data, sfcrs)

    return forwardingGraphs
