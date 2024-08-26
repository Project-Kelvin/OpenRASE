"""
This defines the Neural Network used for genetic encoding.
"""

import json
import random
import pandas as pd
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig

from utils.embedding_graph import traverseVNF
from utils.topology import generateFatTreeTopology


def convertFGstoDF(fgs: "list[EmbeddingGraph]", topology: Topology) -> pd.DataFrame:
    """
    Converts a list of EmbeddingGraphs to a Pandas Dataframe.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of EmbeddingGraphs.
        topology (Topology): the topology.
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
                data.append([fgID, vnfID, pos, hIndex + 1])

        traverseVNF(fg["vnfs"], parseVNF, shouldParseTerminal=False)

    return pd.DataFrame(data, columns=["SFC", "VNF", "Position", "Host"])


def generateEmbeddingGraphs(data: pd.DataFrame, fgs: "list[EmbeddingGraph]", topology: Topology) -> "list[EmbeddingGraph]":
    """
    Generates the Embedding Graphs.

    Parameters:
        data (pd.DataFrame): the data.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.
    """

    noHosts: int = len(topology["hosts"])
    startIndex: "list[int]" = [0]
    endIndex: "list[int]" = [noHosts - 1]

    egs: "list[EmbeddingGraph]" = []
    for index, fg in enumerate(fgs):
        embeddingNotFound: "list[bool]" = [False]

        def parseVNF(vnf: VNF, pos: int, startIndex, endIndex) -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                pos (int): the position.
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

        traverseVNF(fg["vnfs"], parseVNF, startIndex, endIndex, shouldParseTerminal=False)

        if not embeddingNotFound[0]:
            egs.append(fg)

    return egs


topo: Topology = generateFatTreeTopology(4, 1000, 2, 1000)

with open(f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs/forwarding-graphs.json", "r") as file:
    fgs: "list[EmbeddingGraph]" = json.load(file)

    data = convertFGstoDF(fgs, topo)
    cls  =[]
    for _i in range(len(data)):
        cls.append(random.uniform(0, 1))
    data = data.assign(ConfidenceLevel = cls)
    egs = generateEmbeddingGraphs(data, fgs, topo)

    print(egs)
