"""
This defines the function to generate a random individual for the Gaha algorithm.
"""

import copy
import random
from typing import Type
from uuid import uuid4
from dijkstar import Graph, find_path

from algorithms.hybrid.constants.surrogate import BRANCH
from algorithms.hybrid.models.individuals import Individual
from algorithms.mak_ga.models.embedding import EmbeddingData
from algorithms.models.embedding import LinkData
from algorithms.utils.graphs import getVNFsFromFGRs, parseNodes
from constants.topology import SERVER, SFCC
from packages.python.shared.models.embedding_graph import VNF, EmbeddingGraph
from packages.python.shared.models.topology import Topology
from utils.embedding_graph import traverseVNF


def generateRandomIndividual(
    container: Type[Individual],
    fgrs: "list[EmbeddingGraph]",
    topology: Topology,
    rejectionRate: float = 0.05
) -> Individual:
    """
    Generates a random individual for the genetic algorithm.

    Parameters:
        container (Type[Individual]): The type of individual to generate.
        rejectionRate (float): The probability of a VNF being deployed on a host.

    Returns:
        Individual: The generated random individual.
    """

    individual: Individual = container()
    noOfVNFs: int = len(getVNFsFromFGRs(fgrs))
    noOfHosts: int = len(topology["hosts"])
    isValid: bool = False

    while not isValid:
        for _i in range(noOfVNFs):
            host: int = random.randint(1, noOfHosts)
            if random.random() < rejectionRate:
                host = 0
            individual.append(host)

        # decodedIndividual = self.decodePop([individual])[0]
        # data: TimeSFCRequests = self._generateTrafficData(
        #     decodedIndividual[1], isMax=True
        # )
        # MakGAUtils._demandPredictions.cacheResourceDemands(decodedIndividual[1], data)
        # isValid = not isHostConstraintViolated(
        #     decodedIndividual
        # ) and not isLinkConstraintViolated(decodedIndividual)
        isValid = True

    individual.id = uuid4()

    return individual

def convertIndividualToEmbeddingGraphs(individual: Individual, topology: Topology, fgrs: list[EmbeddingGraph], popIndex: int, ignoreVNFInstances: bool = False) -> tuple[
    list[EmbeddingGraph],
    EmbeddingData,
    LinkData,
    dict[str, list[tuple[str, int, int]]],
    int
]:
    """
    Converts a list of integers representing an individual into a list of EmbeddingGraph objects.

    Parameters:
        individual (Individual): The individual to convert.
        popIndex (int): The index of the individual in the population.
        ignoreVNFInstances (bool): Whether to ignore VNF instances.

    Returns:
        tuple[list[EmbeddingGraph], EmbeddingData, LinkData, dict[str, list[tuple[str, int]]]]:
            A tuple containing:
            - A list of EmbeddingGraph objects.
            - An EmbeddingData object containing the embedding data.
            - A LinkData object containing the link data.
            - A dictionary mapping SFC IDs to lists of tuples containing VNF IDs, their instance, and their depths.
            - The index of the individual in the population.
    """

    copiedIndividual = individual.copy()
    nodes: "dict[str, list[str]]" = {}
    embeddingData: EmbeddingData = {}
    vnfData: "dict[str, list[tuple[str, int, int]]]" = {}
    linkData: LinkData = {}
    egs: "list[EmbeddingGraph]" = []
    vnfInstances: dict[str, dict[str, int]] = {}

    def parseVNF(
        vnf: VNF,
        depth: int,
        embeddingNotFound: tuple[bool],
        oldDepth: tuple[int],
        fgr: EmbeddingGraph,
    ) -> None:
        """
        Recursively parses a VNF and its children to create an EmbeddingGraph.

        Parameters:
            vnf (VNF): The VNF object to parse.
            depth (int): The current depth in the recursion.
            embeddingNotFound (tuple[bool]): A tuple indicating whether an embedding was not found.
            oldDepth (tuple[int]): A tuple containing the previous depth.
            fgr (EmbeddingGraph): The current EmbeddingGraph being constructed.
        """

        nonlocal vnfInstances

        if depth != oldDepth[0]:
            oldDepth[0] = depth
            if nodes[fgr["sfcID"]][-1] != SERVER:
                nodes[fgr["sfcID"]].append(BRANCH)

        if embeddingNotFound[0]:
            return

        if "host" in vnf and vnf["host"]["id"] == SERVER:
            nodes[fgr["sfcID"]].append(SERVER)

            return

        hostIndex: int = copiedIndividual.pop(0)

        if hostIndex == 0:
            embeddingNotFound[0] = True

            return

        hostID: str = topology["hosts"][hostIndex - 1]["id"]
        vnf["host"] = {
            "id": hostID,
        }

        if nodes[fgr["sfcID"]][-1] != vnf["host"]["id"]:
            nodes[fgr["sfcID"]].append(vnf["host"]["id"])

        vnfInstance: int = 1
        if (
            fgr["sfcID"] in vnfInstances
            and vnf["vnf"]["id"] in vnfInstances[fgr["sfcID"]]
        ):
            vnfInstances[fgr["sfcID"]][vnf["vnf"]["id"]] += 1
            vnfInstance = vnfInstances[fgr["sfcID"]][vnf["vnf"]["id"]]
        elif fgr["sfcID"] in vnfInstances:
            vnfInstances[fgr["sfcID"]][vnf["vnf"]["id"]] = 1
        else:
            vnfInstances[fgr["sfcID"]] = {vnf["vnf"]["id"]: 1}

        if fgr["sfcID"] not in vnfData:
            if ignoreVNFInstances:
                vnfData[fgr["sfcID"]] = [(vnf["vnf"]["id"], depth)]
            else:
                vnfData[fgr["sfcID"]] = [(vnf["vnf"]["id"], vnfInstance, depth)]
        else:
            if ignoreVNFInstances:
                vnfData[fgr["sfcID"]].append((vnf["vnf"]["id"], depth))
            else:
                vnfData[fgr["sfcID"]].append((vnf["vnf"]["id"], vnfInstance, depth))

        if vnf["host"]["id"] in embeddingData:
            if fgr["sfcID"] in embeddingData[vnf["host"]["id"]]:
                if ignoreVNFInstances:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]].append(
                        [vnf["vnf"]["id"], depth]
                    )
                else:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]].append(
                        [vnf["vnf"]["id"], vnfInstance, depth]
                    )
            else:
                if ignoreVNFInstances:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]] = [
                        [vnf["vnf"]["id"], depth]
                    ]
                else:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]] = [
                        [vnf["vnf"]["id"], vnfInstance, depth]
                    ]
        else:
            if ignoreVNFInstances:
                embeddingData[vnf["host"]["id"]] = {
                    fgr["sfcID"]: [[vnf["vnf"]["id"], depth]]
                }
            else:
                embeddingData[vnf["host"]["id"]] = {
                    fgr["sfcID"]: [[vnf["vnf"]["id"], vnfInstance, depth]]
                }

    for index, fgr in enumerate(fgrs):
        copiedFGR: EmbeddingGraph = copy.deepcopy(fgr)
        embeddingNotFound = [False]
        vnfs: VNF = copiedFGR["vnfs"]
        copiedFGR["sfcID"] = (
            copiedFGR["sfcrID"] if "sfcrID" in copiedFGR else f"sfc{index}"
        )
        nodes[copiedFGR["sfcID"]] = [SFCC]
        oldDepth: tuple[int] = [1]

        traverseVNF(vnfs, parseVNF, embeddingNotFound, oldDepth, copiedFGR)

        if not embeddingNotFound[0]:
            if "sfcrID" in copiedFGR:
                del copiedFGR["sfcrID"]

            graph = Graph()
            paths: "dict[str, list[str]]" = {}
            eg: EmbeddingGraph = copy.deepcopy(copiedFGR)

            if "links" not in eg:
                eg["links"] = []

            for link in topology["links"]:
                graph.add_edge(
                    link["source"],
                    link["destination"],
                    (
                        link["delay"]
                        if "delay" in link and link["delay"] is not None
                        else 1
                    ),
                )
                graph.add_edge(
                    link["destination"],
                    link["source"],
                    (
                        link["delay"]
                        if "delay" in link and link["delay"] is not None
                        else 1
                    ),
                )

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
                            if eg["sfcID"] in linkData[f"{path[p]}-{path[p + 1]}"]:
                                pathData: tuple[float, float] = linkData[
                                    f"{path[p]}-{path[p + 1]}"
                                ][eg["sfcID"]]
                                divisors = pathData[0] + (1 / divisor)
                                delay = pathData[1] + linkDelay
                                linkData[f"{path[p]}-{path[p + 1]}"][
                                    eg["sfcID"]
                                ] = (
                                    divisors,
                                    delay,
                                )
                            else:
                                linkData[f"{path[p]}-{path[p + 1]}"][
                                    eg["sfcID"]
                                ] = (
                                    (1 / divisor),
                                    linkDelay,
                                )
                        elif f"{path[p + 1]}-{path[p]}" in linkData:
                            if eg["sfcID"] in linkData[f"{path[p + 1]}-{path[p]}"]:
                                pathData: tuple[float, float] = linkData[
                                    f"{path[p + 1]}-{path[p]}"
                                ][eg["sfcID"]]
                                divisors = pathData[0] + (1 / divisor)
                                delay = pathData[1] + linkDelay
                                linkData[f"{path[p + 1]}-{path[p]}"][
                                    eg["sfcID"]
                                ] = (
                                    divisors,
                                    delay,
                                )
                            else:
                                linkData[f"{path[p + 1]}-{path[p]}"][
                                    eg["sfcID"]
                                ] = (
                                    1 / divisor,
                                    linkDelay,
                                )
                        else:
                            linkData[f"{path[p]}-{path[p + 1]}"] = {
                                eg["sfcID"]: (1 / divisor, linkDelay)
                            }

            egs.append(eg)

    return (
        egs,
        embeddingData,
        linkData,
        vnfData,
        popIndex,
    )
