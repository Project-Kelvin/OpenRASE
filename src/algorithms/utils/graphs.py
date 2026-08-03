"""
Defines util functions used to process graphs.
"""


import copy
from typing import Tuple

from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.utils.config import getConfig
from algorithms.hybrid.constants.surrogate import BRANCH
from constants.topology import SERVER
from utils.embedding_graph import traverseVNF


def getVNFsFromFGRs(fgrs: "list[EmbeddingGraph]") -> "list[str]":
    """
    Get the VNFs from the SFC Request.

    Parameters:
        fgrs (list[EmbeddingGraph]): the FG Requests.

    Returns:
        list[str]: the VNFs.
    """

    vnfs: "list[tuple[str, int]]" = []

    def parseVNF(vnf: VNF, depth: int, vnfs: "list[str]") -> None:
        """
        Parse the VNF.

        Parameters:
            vnf (VNF): the VNF.
            depth (int): the depth.

        Returns:
            None
        """

        vnfs.append((vnf["vnf"]["id"], depth))

    for fgr in fgrs:
        traverseVNF(fgr["vnfs"], parseVNF, vnfs, shouldParseTerminal=False)

    return vnfs


def parseNodes(nodes: "list[str]") -> "tuple[list[list[str]], list[int]]":
    """
    Parses the nodes.

    Parameters:
        nodes (list[str]): the nodes.

    Returns:
        Tuple[list[list[str]], list[int]]: the parsed nodes, the parsed divisors.
    """

    parsedNodes: "list[list[str]]" = []
    roots: "list[list[str]]" = []
    branch: "list[str]" = []
    connectingNode: str = None
    currentDivisor: int = 1
    divisors: "list[int]" = []
    parsedDivisors: "list[int]" = []

    for node in nodes:
        if node == BRANCH:
            roots.append(branch[:])
            parsedNodes.append(branch[:])
            parsedDivisors.append(currentDivisor)
            currentDivisor *= 2
            divisors.append(currentDivisor)
            connectingNode = branch[-1]
            branch = []
        elif node == SERVER:
            if connectingNode:
                parsedNodes.append([connectingNode, node])
                parsedDivisors.append(currentDivisor)
                connectingNode = None
            else:
                branch.append(node)
                parsedNodes.append(branch[:])
                parsedDivisors.append(currentDivisor)
                branch = []
            if len(roots) > 0:
                lastRoot: "list[str]" = roots.pop()
                currentDivisor = divisors.pop()
                connectingNode = lastRoot[-1]
        else:
            if connectingNode:
                parsedNodes.append([connectingNode, node])
                parsedDivisors.append(currentDivisor)
                connectingNode = None
            branch.append(node)

    return parsedNodes, parsedDivisors

def convertSFCRsToEGs(sfcrs: list[SFCRequest]) -> list[EmbeddingGraph]:
        """
        Converts a list of SFCRequests to a list of EmbeddingGraphs.

        Parameters:
            sfcrs (list[SFCRequest]): A list of SFCRequests.

        Returns:
            list[EmbeddingGraph]: A list of EmbeddingGraphs.
        """

        egs: "list[EmbeddingGraph]" = []
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]" = {}
        splitters: "list[str]" = getConfig()["vnfs"]["splitters"]

        for sfcr in sfcrs:
            sfcrID: str = sfcr["sfcrID"]
            sortedVNFs: "list[str]" = copy.deepcopy(sfcr["vnfs"])
            forwardingGraph: EmbeddingGraph = {"sfcID": sfcrID, "vnfs": {}}
            oldDepth: int = 1
            depth: int = 1
            vnfDict: VNF = forwardingGraph["vnfs"]

            def addVNF(vnfs: "list[str]", vnfDict: VNF, depth: int) -> None:
                """
                Adds VNF to the EmbeddingGraph.

                Parameters:
                    vnfs (list[str]): the VNFs.
                    vnfDict (VNF): the VNF dictionary.
                    depth (int): the depth of the VNF.

                Returns:
                    None
                """

                nonlocal oldDepth, embeddingData, splitters


                if len(vnfs) == 0:
                    vnfDict["host"] = {"id": SERVER}
                    vnfDict["next"] = TERMINAL

                    return

                vnf: str = vnfs.pop(0)
                splitter: bool = vnf in splitters
                vnfDict["next"] = [{}, {}] if splitter else {}

                vnfDict["vnf"] = {"id": vnf}

                if splitter:
                    depth += 1
                    for i in range(2):
                        addVNF(copy.deepcopy(vnfs), vnfDict["next"][i], depth)
                else:
                    addVNF(vnfs, vnfDict["next"], depth)

            addVNF(sortedVNFs, vnfDict, depth)
            egs.append(forwardingGraph)

        return egs
