"""
This defines the GA algorithms developed by Mohammad Ali Khoshkholghi.
"""

import copy
import random
import numpy as np
from shared.models.embedding_graph import VNF, EmbeddingGraph, ForwardingLink
from shared.models.topology import Link, Topology
from dijkstar import Graph, find_path
from shared.models.traffic_design import TrafficDesign
from algorithms.ga_dijkstra_algorithm.ga_utils import getVNFsFromFGRs, parseNodes
from algorithms.models.embedding import DecodedIndividual, EmbeddingData, LinkData
from algorithms.surrogacy.constants.surrogate import BRANCH
from algorithms.surrogacy.models.traffic import TimeSFCRequests
from algorithms.surrogacy.utils.demand_predictions import DemandPredictions
from algorithms.surrogacy.utils.scorer import Scorer
from constants.topology import SERVER, SFCC
from models.calibrate import ResourceDemand
from utils.data import getAvailableCPUAndMemory
from utils.embedding_graph import traverseVNF
from utils.traffic_design import getTrafficDesignRate


def generateRandomIndividual(container: list, fgrs: list[EmbeddingGraph], trafficDesign: list[TrafficDesign], topology: Topology, demandPrediction: DemandPredictions) -> list[int]:
    """
    Generates a random individual for the genetic algorithm.

    Parameters:
        container (list): The container to hold the individual.
        fgrs (list[EmbeddingGraph]): A list of forwarding graph requests.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.
        topology (Topology): The topology object containing the network information.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.

    Returns:
        list[int]: A list of integers representing the random individual.
    """

    individual: list[int] = container()

    noOfVNFs: int = len(getVNFsFromFGRs(fgrs))
    noOfHosts: int = len(topology["hosts"])
    isValid: bool = False

    while not isValid:
        for _i in range(noOfVNFs):
            individual.append(random.randint(0, noOfHosts))

        decodedIndividual = decodePop([individual], fgrs, topology)[0]
        data: TimeSFCRequests = generateTrafficData(trafficDesign, decodedIndividual[1], isMax=True)
        demandPrediction.cacheResourceDemands(decodedIndividual[1], data)
        isValid = (
            not isHostConstraintViolated(
                decodedIndividual, topology, demandPrediction, trafficDesign
            )
            and not isLinkConstraintViolated(
                decodedIndividual, topology, trafficDesign
            )
        )

    return individual


def convertIndividualToEmbeddingGraphs(
    individual: list[int], fgrs: list[EmbeddingGraph], topology: Topology
) -> tuple[
    list[EmbeddingGraph], EmbeddingData, LinkData, dict[str, list[tuple[str, int]]]
]:
    """
    Converts a list of integers representing an individual into a list of EmbeddingGraph objects.

    Parameters:
        individual (list[int]): A list of integers where each integer represents the index of a host.
        fgrs (list[EmbeddingGraph]): A list of EmbeddingGraph objects corresponding to the VNFs.
        topology (Topology): The topology object containing the network information.

    Returns:
        tuple[list[EmbeddingGraph], EmbeddingData, LinkData, dict[str, list[tuple[str, int]]]]:
            A tuple containing:
            - A list of EmbeddingGraph objects.
            - An EmbeddingData object containing the embedding data.
            - A LinkData object containing the link data.
            - A dictionary mapping SFC IDs to lists of tuples containing VNF IDs and their depths.
    """

    copiedIndividual = individual.copy()
    nodes: "dict[str, list[str]]" = {}
    embeddingData: EmbeddingData = {}
    vnfData: "dict[str, list[tuple[str, int]]]" = {}
    linkData: LinkData = {}
    egs: "list[EmbeddingGraph]" = []

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

        if depth != oldDepth[0]:
            oldDepth[0] = depth
            if nodes[fgr["sfcID"]][-1] != SERVER:
                nodes[fgr["sfcID"]].append(BRANCH)

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

        if fgr["sfcID"] not in vnfData:
            vnfData[fgr["sfcID"]] = [(vnf["vnf"]["id"], depth)]
        else:
            vnfData[fgr["sfcID"]].append((vnf["vnf"]["id"], depth))

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

    for index, fgr in enumerate(fgrs):
        embeddingNotFound = [False]
        vnfs: VNF = fgr["vnfs"]
        fgr["sfcID"] = fgr["sfcrID"] if "sfcrID" in fgr else f"sfc{index}"
        nodes[fgr["sfcID"]] = [SFCC]
        oldDepth: tuple[int] = [1]

        traverseVNF(vnfs, parseVNF, embeddingNotFound, oldDepth, fgr)

        if not embeddingNotFound[0]:
            if "sfcrID" in fgr:
                del fgr["sfcrID"]

            graph = Graph()
            nodePair: "list[str]" = []
            eg: EmbeddingGraph = copy.deepcopy(fgr)

            if "links" not in eg:
                eg["links"] = []

            for link in topology["links"]:
                graph.add_edge(link["source"], link["destination"], link["bandwidth"])
                graph.add_edge(link["destination"], link["source"], link["bandwidth"])

            sfcNodes, sfcDivisors = parseNodes(nodes[eg["sfcID"]])
            for nodeList, divisor in zip(sfcNodes, sfcDivisors):
                for i in range(len(nodeList) - 1):
                    if nodeList[i] == nodeList[i + 1]:
                        continue
                    srcDst: str = f"{nodeList[i]}-{nodeList[i + 1]}"
                    dstSrc: str = f"{nodeList[i + 1]}-{nodeList[i]}"
                    if srcDst not in nodePair and dstSrc not in nodePair:
                        nodePair.append(srcDst)
                        nodePair.append(dstSrc)
                        try:
                            path = find_path(graph, nodeList[i], nodeList[i + 1])
                        except Exception as e:
                            raise (e)

                        for p in range(len(path.nodes) - 1):
                            if f"{path.nodes[p]}-{path.nodes[p + 1]}" in linkData:
                                if (
                                    eg["sfcID"]
                                    in linkData[f"{path.nodes[p]}-{path.nodes[p + 1]}"]
                                ):
                                    linkData[f"{path.nodes[p]}-{path.nodes[p + 1]}"][
                                        eg["sfcID"]
                                    ] += (1 / divisor)
                                else:
                                    linkData[f"{path.nodes[p]}-{path.nodes[p + 1]}"][
                                        eg["sfcID"]
                                    ] = (1 / divisor)
                            elif f"{path.nodes[p + 1]}-{path.nodes[p]}" in linkData:
                                if (
                                    eg["sfcID"]
                                    in linkData[f"{path.nodes[p + 1]}-{path.nodes[p]}"]
                                ):
                                    linkData[f"{path.nodes[p + 1]}-{path.nodes[p]}"][
                                        eg["sfcID"]
                                    ] += (1 / divisor)
                                else:
                                    linkData[f"{path.nodes[p + 1]}-{path.nodes[p]}"][
                                        eg["sfcID"]
                                    ] = (1 / divisor)
                                linkData[f"{path.nodes[p + 1]}-{path.nodes[p]}"][
                                    eg["sfcID"]
                                ] += (1 / divisor)
                            else:
                                linkData[f"{path.nodes[p]}-{path.nodes[p + 1]}"] = {
                                    eg["sfcID"]: 1 / divisor
                                }
                        eg["links"].append(
                            {
                                "source": {"id": path.nodes[0]},
                                "destination": {"id": path.nodes[-1]},
                                "links": path.nodes[1:-1],
                                "divisor": divisor,
                            }
                        )

            egs.append(eg)

    return egs, embeddingData, linkData, vnfData


def decodePop(
    pop: list[list[int]], fgrs: list[EmbeddingGraph], topology: Topology
) -> list[
    tuple[
        int,
        list[EmbeddingGraph],
        EmbeddingData,
        LinkData,
        float,
        dict[str, list[tuple[str, int]]],
    ]
]:
    """
    Decodes a population of individuals into EmbeddingGraph objects and calculates the total cost.

    Parameters:
        pop (list[list[int]]): A list of individuals, where each individual is a list of integers.
        fgrs (list[EmbeddingGraph]): A list of EmbeddingGraph objects corresponding to the VNFs.
        topology (Topology): The topology object containing the network information.

    Returns:
        list[tuple[int, list[EmbeddingGraph], EmbeddingData, LinkData, float, dict[str, list[tuple[str, int]]]]]:A list containing a tuple that consists of the index, embedding graphs, embedding data, link data, acceptance ratio, and VNF data for each individual.
    """

    decodedPop: list[DecodedIndividual] = []

    for index, individual in enumerate(pop):
        egs, embeddingData, linkData, vnfData = convertIndividualToEmbeddingGraphs(
            individual, fgrs, topology
        )

        acceptanceRatio: float = len(egs) / len(fgrs) if fgrs else 0.0

        decodedPop.append(
            (
                index,
                egs,
                EmbeddingData(embeddingData),
                LinkData(linkData),
                acceptanceRatio,
                vnfData,
            )
        )

    return decodedPop



def generateTrafficData(
    trafficDesign: list[TrafficDesign], egs: list[EmbeddingGraph], isMax: bool = False
) -> TimeSFCRequests:
    """
    Generates traffic data from a traffic design.

    Parameters:
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.
        egs (list[EmbeddingGraph]): A list of EmbeddingGraph objects.
        isMax (bool): If True, returns the maximum requests per second; otherwise, returns the average.

    Returns:
        TimeSFCRequests: A TimeSFCRequests object containing the traffic data.
    """

    data: TimeSFCRequests = []

    if isMax:
        maxTarget: float = max(
            trafficDesign[0], key=lambda x: x["target"]
        )
        timeData: dict[str, float] = {
            eg["sfcID"]: maxTarget for eg in egs
        }
        data.append(timeData)

        return data
    avgReqs: float = np.mean(getTrafficDesignRate(trafficDesign[0]))
    data = [ {eg["sfcID"]: avgReqs for eg in egs} ]

    return data

def isVNFInHost(
    sfcID: str, vnfID: str, hostID: str, embeddingData: EmbeddingData
) -> bool:
    """
    Checks if a VNF is embedded in a specific host.

    Parameters:
        sfcID (str): The ID of the SFC.
        vnfID (str): The ID of the VNF.
        hostID (str): The ID of the host.
        embeddingData (EmbeddingData): The embedding data containing host information.

    Returns:
        bool: True if the VNF is in the host, False otherwise.
    """

    return (
        hostID in embeddingData
        and sfcID in embeddingData[hostID]
        and any(vnf[0] == vnfID for vnf in embeddingData[hostID][sfcID])
    )


def getProcessingDelay(
    data: dict[str, float],
    decodedIndividual: DecodedIndividual,
    topology: Topology,
    demandPrediction: DemandPredictions,
) -> float:
    """
    Calculates the processing delay for a decoded individual based on the topology.

    Parameters:
        data (dict[str, float]): The traffic data containing requests for each SFC.
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.

    Returns:
        float: The total CPU cost for the individual.
    """

    egs: list[EmbeddingGraph] = decodedIndividual[1]

    cpuCost: float = 0.0
    for eg in egs:
        for host in topology["hosts"]:
            for _ in range(len(decodedIndividual[5][eg["sfcID"]])):
                serverCPU, _ = getAvailableCPUAndMemory()
                cpuAvailable: float = host["cpu"] if host["cpu"] is not None else serverCPU
                totalVNFDemand: float = 0.0

                for vnf, depth in decodedIndividual[5][eg["sfcID"]]:
                    vnfDemand: float = 0.0
                    divisor: int = 2 ** (depth - 1)

                    if isVNFInHost(eg["sfcID"], vnf, host["id"], decodedIndividual[2]):
                        vnfDemand = demandPrediction.getDemand(
                            vnf, (data[eg["sfcID"]] / divisor) if eg["sfcID"] in data else 0
                        )["cpu"]

                    totalVNFDemand += vnfDemand

                cpuCost += 1 / (cpuAvailable - totalVNFDemand)

    return cpuCost


def isVirtualLinkInPhysicalLink(
    topoLink: Link,
    egLink: ForwardingLink
) -> bool:
    """
    Checks if a virtual link is present in a physical link.

    Parameters:
        topoLink (Link): The physical link in the topology.
        egLink (ForwardingLink): The virtual link in the embedding graph.

    Returns:
        bool: True if the virtual link is present in the physical link, False otherwise.
    """

    links: list[str] = [egLink["source"]["id"]]
    links.extend(egLink["links"])
    links.append(egLink["destination"]["id"])
    for src, dest in zip(links[::2], links[1::2]):
        if (src == topoLink["source"] and
            dest == topoLink["destination"]) or (
            src == topoLink["destination"] and
            dest == topoLink["source"]):
            return True

    return False

def getPropagationDelay(
    decodedIndividual: DecodedIndividual,
    topology: Topology
) -> float:
    """
    Calculates the propagation delay for a decoded individual based on the topology.

    Parameters:
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.

    Returns:
        float: The total propagation delay for the individual.
    """

    egs: list[EmbeddingGraph] = decodedIndividual[1]
    propagationDelayOfLink: float = 1.0

    propagationDelay: float = 0.0
    for eg in egs:
        for topoLink in topology["links"]:
            for link in eg["links"]:
                if isVirtualLinkInPhysicalLink(topoLink, link):
                    propagationDelay += propagationDelayOfLink

    return propagationDelay

def getQueueDelay(
    data: dict[str, float],
    decodedIndividual: DecodedIndividual,
    topology: Topology,
) -> float:
    """
    Calculates the queue delay for a decoded individual based on the topology.

    Parameters:
        data (dict[str, float]): The traffic data containing requests for each SFC.
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.

    Returns:
        float: The total queue delay for the individual.
    """
    egs: list[EmbeddingGraph] = decodedIndividual[1]
    queueDelay: float = 0.0

    for eg in egs:
        for topoLink in topology["links"]:
            for _ in range(len(eg["links"])):
                totalLinkDemand: float = 0.0

                for link in eg["links"]:
                    linkDemand: float = 0.0
                    if isVirtualLinkInPhysicalLink(topoLink, link):
                        linkDemand = (data[eg["sfcID"]] / link["divisor"]) if eg["sfcID"] in data else 0

                    totalLinkDemand += linkDemand

                queueDelay += 1/(topoLink["bandwidth"] - totalLinkDemand)

    return queueDelay


def getVirtualisationDelay(
    decodedIndividual: DecodedIndividual,
    topology: Topology,
) -> float:
    """
    Calculates the virtualisation delay for a decoded individual based on the topology.

    Parameters:
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.

    Returns:
        float: The total CPU cost for the individual.
    """

    egs: list[EmbeddingGraph] = decodedIndividual[1]
    hostVirtualisationDelay: float = 1.0

    virtualisationDelay: float = 0.0
    for eg in egs:
        for host in topology["hosts"]:
            for vnf, _depth in decodedIndividual[5][eg["sfcID"]]:
                if isVNFInHost(eg["sfcID"], vnf, host["id"], decodedIndividual[2]):
                    virtualisationDelay += hostVirtualisationDelay

    return virtualisationDelay


def isHostConstraintViolated(
    decodedIndividual: DecodedIndividual,
    topology: Topology,
    demandPrediction: DemandPredictions,
    trafficDesign: list[TrafficDesign],
) -> bool:
    """
    Checks if the host constraints are violated for a decoded individual based on the topology.

    Parameters:
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.

    Returns:
        float: The total CPU cost for the individual.
    """
    egs: list[EmbeddingGraph] = decodedIndividual[1]
    data: TimeSFCRequests = generateTrafficData(trafficDesign, egs, isMax=True)

    scores: dict[str, ResourceDemand] = Scorer.getHostScores(data, topology, decodedIndividual[2], demandPrediction)
    maxCPU: float = max(scores.values(), key=lambda score: score["cpu"] )["cpu"]
    maxMemory: float = max(scores.values(), key=lambda score: score["memory"] )["memory"]

    return maxCPU > 1.0 or maxMemory > 1.0

def isLinkConstraintViolated(
    decodedIndividual: DecodedIndividual,
    topology: Topology,
    trafficDesign: list[TrafficDesign],
) -> bool:
    """
    Checks if the link constraints are violated for a decoded individual based on the topology.

    Parameters:
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.

    Returns:
        bool: True if the link constraints are violated, False otherwise.
    """
    egs: list[EmbeddingGraph] = decodedIndividual[1]
    linkData: LinkData = decodedIndividual[3]
    data: TimeSFCRequests = generateTrafficData(trafficDesign, egs, isMax=True)

    linkRequestData: dict[str, float] = {}
    for eg in egs:
        for egLink in eg["links"]:
            links: "list[str]" = [egLink["source"]["id"]]
            links.extend(egLink["links"])
            links.append(egLink["destination"]["id"])
            reqps: float = data[eg["sfcID"]] if eg["sfcID"] in data else 0

            for linkIndex in range(len(links) - 1):
                source: str = links[linkIndex]
                destination: str = links[linkIndex + 1]

                if f"{source}-{destination}" in linkData:
                    for _key, factor in linkData[f"{source}-{destination}"].items():
                        if f"{source}-{destination}" in linkRequestData:
                            linkRequestData[f"{source}-{destination}"] += factor * reqps
                        else:
                            linkRequestData[f"{source}-{destination}"] = factor * reqps
                elif f"{destination}-{source}" in linkData:
                    for _key, factor in linkData[f"{destination}-{source}"].items():
                        if f"{destination}-{source}" in linkRequestData:
                            linkRequestData[f"{destination}-{source}"] += factor * reqps
                        else:
                            linkRequestData[f"{destination}-{source}"] = factor * reqps

    for key, linkRequests in linkRequestData.items():
        source, destination = key.split("-")
        bandwidth: float = [
            link["bandwidth"]
            for link in topology["links"]
            if (link["source"] == source and link["destination"] == destination)
            or (link["source"] == destination and link["destination"] == source)
        ][0]

        if linkRequests > bandwidth:
            return True

    return False

def getTotalDelay(
        decodedIndividual: DecodedIndividual,
        topology: Topology,
        demandPrediction: DemandPredictions,
        trafficDesign: list[TrafficDesign],
) -> float:
    """
    Calculates the total delay for a decoded individual based on the topology.

    Parameters:
        decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.
        topology (Topology): The topology object containing the network information.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.

    Returns:
        float: The total delay for the individual.
    """

    data: TimeSFCRequests = generateTrafficData(trafficDesign, decodedIndividual[1])
    propagationDelay: float = getPropagationDelay(decodedIndividual, topology)
    virtualisationDelay: float = getVirtualisationDelay(decodedIndividual, topology)
    processingDelays: list[float]= []
    queueDelays: list[float] = []

    for time in data:
        processingDelay: float = getProcessingDelay(
            time, decodedIndividual, topology, demandPrediction
        )
        queueDelay: float = getQueueDelay(
            time, decodedIndividual, topology
        )

        processingDelays.append(processingDelay)
        queueDelays.append(queueDelay)

    meanProcessingDelay: float = np.mean(processingDelays) if processingDelays else 0.0
    meanQueueDelay: float = np.mean(queueDelays) if queueDelays else 0.0

    return meanProcessingDelay + meanQueueDelay + propagationDelay + virtualisationDelay


def cacheDemand(pop: list[DecodedIndividual], demandPrediction: DemandPredictions, trafficDesign: list[TrafficDesign]) -> None:
    """
    Caches the resource demands for each embedding graph based on the provided data.

    Parameters:
        pop (list[DecodedIndividual]): List of decoded individuals.
        demandPrediction (DemandPredictions): The demand predictions object containing resource demands.
        trafficDesign (list[TrafficDesign]): A list of traffic design objects.

    Returns:
        None
    """

    egs: list[EmbeddingGraph] = []
    for ind in pop:
        egs.extend(ind[1])

    data: TimeSFCRequests = generateTrafficData(trafficDesign, ind[1])
    maxData: TimeSFCRequests = generateTrafficData(trafficDesign, ind[1], isMax=True)

    demandPrediction.cacheResourceDemands(egs, data + maxData)
