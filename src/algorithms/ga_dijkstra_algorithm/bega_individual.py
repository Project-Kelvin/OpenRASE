
"""
This defines the BEGA individual generation and conversion functions.
"""

import copy
import random
from typing import Tuple, Type
from uuid import uuid4
from dijkstar import Graph, find_path

from algorithms.hybrid.constants.surrogate import BRANCH
from algorithms.hybrid.models.individuals import Individual
from algorithms.mak_ga.models.embedding import EmbeddingData
from algorithms.models.embedding import LinkData
from algorithms.utils.graphs import convertSFCRsToEGs, getVNFsFromFGRs, parseNodes
from constants.topology import SERVER, SFCC
from packages.python.shared.models.embedding_graph import VNF, EmbeddingGraph
from packages.python.shared.models.sfc_request import SFCRequest
from packages.python.shared.models.topology import Link, Topology
from utils.embedding_graph import traverseVNF
from utils.tui import TUI


def generateRandomIndividual(
    container: Type[Individual], topo: Topology, sfcrs: "list[SFCRequest]", rejectionrate: float = 0.05
) -> Individual:
    """
    Generate a random individual.

    Parameters:
        container (Type[Individual]): the container type for the individual.
        topo (Topology): the topology.
        sfcrs (list[SFCRequest]): the SFC Requests.
        rejectionrate (float): the probability of a VNF being deployed on a host.

    Returns:
       Individual: the random individual.
    """

    individual: Individual = container()

    fgrs: list[EmbeddingGraph] = convertSFCRsToEGs(sfcrs)
    vnfs: "list[str]" = getVNFsFromFGRs(fgrs)
    noOfVNFs: int = len(vnfs)
    noOfHosts: int = len(topo["hosts"])

    for _ in range(noOfVNFs):
        item: "list[int]" = [0] * noOfHosts
        if random.random() >= rejectionrate:
            item[random.randint(0, noOfHosts - 1)] = 1
        individual.append(item)

    individual.id = uuid4()

    return individual


def convertIndividualToEmbeddingGraph(
    individual: "list[list[int]]", sfcrs: "list[SFCRequest]", topology: Topology, popIndex: int, rejectionRate: float = 0.05
) -> "Tuple[list[EmbeddingGraph], EmbeddingData, LinkData, int]":
    """
    Convert individual to an embedding graph.

    Parameters:
        individual (list[list[int]]): the individual to convert.
        sfcrs (list[SFCRequest]): The SFC Requests.
        topology (Topology): The Topology.
        popIndex (int): The index of the embedding graph in the population.
        rejectionRate (float): The probability of a VNF being deployed on a host.

    Returns:
        tuple[list[EmbeddingGraph], EmbeddingData, LinkData, popIndex]: the embedding graph, the embedding data, the link data, and the index.
    """

    egs: "list[EmbeddingGraph]" = []
    offset: "list[int]" = [0]
    nodes: "dict[str, list[str]]" = {}
    embeddingData: EmbeddingData = {}
    linkData: LinkData = {}
    copiedFGRs: "list[EmbeddingGraph]" = copy.deepcopy(convertSFCRsToEGs(sfcrs))

    for index, fgr in enumerate(copiedFGRs):
        vnfs: VNF = fgr["vnfs"]
        embeddingNotFound: "list[bool]" = [False]
        fgr["sfcID"] = fgr["sfcrID"] if "sfcrID" in fgr else f"sfc{index}"
        nodes[fgr["sfcID"]] = [SFCC]
        oldDepth: int = 1

        def parseVNF(
            vnf: VNF, depth: int, embeddingNotFound: "list[bool]", offset: "list[int]"
        ) -> None:
            """
            Parse the VNF.

            Parameters:
                vnf (VNF): the VNF.
                depth (int): the depth.
                embeddingNotFound (list[bool]): the embedding not found.
                offset (list[int]): the offset.

            Returns:
                None
            """

            nonlocal oldDepth

            if depth != oldDepth:
                oldDepth = depth
                if nodes[fgr["sfcID"]][-1] != SERVER:
                    # pylint: disable=cell-var-from-loop
                    nodes[fgr["sfcID"]].append(BRANCH)

            if embeddingNotFound[0]:
                if "host" in vnf and vnf["host"]["id"] == SERVER:
                    return

                offset[0] = offset[0] + 1

                return

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                # pylint: disable=cell-var-from-loop
                nodes[fgr["sfcID"]].append(SERVER)

                return

            else:
                if 1 in individual[offset[0]]:
                    vnf["host"] = {"id": f"h{individual[offset[0]].index(1) + 1}"}
                    # pylint: disable=cell-var-from-loop
                    if nodes[fgr["sfcID"]][-1] != vnf["host"]["id"]:
                        # pylint: disable=cell-var-from-loop
                        nodes[fgr["sfcID"]].append(vnf["host"]["id"])
                    offset[0] = offset[0] + 1

                    if vnf["host"]["id"] in embeddingData:
                        if fgr["sfcID"] in embeddingData[vnf["host"]["id"]]:
                            embeddingData[vnf["host"]["id"]][fgr["sfcID"]].append(
                                [vnf["vnf"]["id"], depth]
                            )
                        else:
                            embeddingData[vnf["host"]["id"]][fgr["sfcID"]] = [
                                [vnf["vnf"]["id"], depth]
                            ]
                    else:
                        embeddingData[vnf["host"]["id"]] = {
                            fgr["sfcID"]: [[vnf["vnf"]["id"], depth]]
                        }
                else:
                    embeddingNotFound[0] = True
                    offset[0] = offset[0] + 1

        traverseVNF(vnfs, parseVNF, embeddingNotFound, offset)
        if not embeddingNotFound[0]:
            if "sfcrID" in fgr:
                del fgr["sfcrID"]

            graph = Graph()
            paths: "dict[str, list[str]]" = {}
            eg: EmbeddingGraph = copy.deepcopy(fgr)

            if "links" not in eg:
                eg["links"] = []

            for link in topology["links"]:
                graph.add_edge(link["source"], link["destination"], link["delay"] if "delay" in link and link["delay"] is not None else 1)
                graph.add_edge(link["destination"], link["source"], link["delay"] if "delay" in link and link["delay"] is not None else 1)

            sfcNodes, sfcDivisors = parseNodes(nodes[eg["sfcID"]])
            for nodeList, divisor in zip(sfcNodes, sfcDivisors):
                for i in range(len(nodeList) - 1):
                    if nodeList[i] == nodeList[i + 1]:
                        continue
                    srcDst: str = f"{nodeList[i]}-{nodeList[i + 1]}"
                    dstSrc: str = f"{nodeList[i + 1]}-{nodeList[i]}"
                    if srcDst not in paths and dstSrc not in paths:
                        try:
                            path = find_path(graph, nodeList[i], nodeList[i + 1])
                            paths.update({srcDst: path.nodes})
                        except Exception as e:
                            TUI.appendToSolverLog(f"Error: {e}")
                            continue

                        eg["links"].append(
                            {
                                "source": {"id": path.nodes[0]},
                                "destination": {"id": path.nodes[-1]},
                                "links": path.nodes[1:-1],
                            }
                        )
                    path = paths[srcDst] if srcDst in paths else paths[dstSrc]
                    for p in range(len(path) - 1):
                        link: Link = [
                            topoLink
                            for topoLink in topology["links"]
                            if (
                                topoLink["source"] == path[p]
                                and topoLink["destination"] == path[p + 1]
                            )
                            or (
                                topoLink["source"] == path[p + 1]
                                and topoLink["destination"] == path[p]
                            )
                        ][0]
                        linkDelay: float = (
                            (link["delay"] / divisor)
                            if "delay" in link and link["delay"] is not None
                            else 0
                        )
                        if f"{path[p]}-{path[p + 1]}" in linkData:
                            if (
                                eg["sfcID"]
                                in linkData[f"{path[p]}-{path[p + 1]}"]
                            ):
                                pathData: tuple[float, float] = linkData[
                                    f"{path[p]}-{path[p + 1]}"
                                ][eg["sfcID"]]
                                divisors = pathData[0] + (1 / divisor)
                                delay = pathData[1] + linkDelay
                                linkData[f"{path[p]}-{path[p + 1]}"][
                                    eg["sfcID"]
                                ] = (divisors, delay)
                            else:
                                linkData[f"{path[p]}-{path[p + 1]}"][
                                    eg["sfcID"]
                                ] = ((1 / divisor), linkDelay)
                        elif f"{path[p + 1]}-{path[p]}" in linkData:
                            if (
                                eg["sfcID"]
                                in linkData[f"{path[p + 1]}-{path[p]}"]
                            ):
                                pathData: tuple[float, float] = linkData[
                                    f"{path[p + 1]}-{path[p]}"
                                ][eg["sfcID"]]
                                divisors = pathData[0] + (1 / divisor)
                                delay = pathData[1] + linkDelay
                                linkData[f"{path[p + 1]}-{path[p]}"][
                                    eg["sfcID"]
                                ] = (divisors, delay)
                            else:
                                linkData[f"{path[p + 1]}-{path[p]}"][
                                    eg["sfcID"]
                                ] = (1 / divisor, linkDelay)
                        else:
                            linkData[f"{path[p]}-{path[p + 1]}"] = {
                                eg["sfcID"]: (1 / divisor, linkDelay)
                            }

            egs.append(eg)

    return egs, embeddingData, linkData, popIndex
