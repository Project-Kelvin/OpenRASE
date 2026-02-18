"""
This defines the util class used by the GA algorithm developed by Mohammad Ali Khoshkholghi.
"""

import copy
import random
import numpy as np
from algorithms.mak_ga.models.embedding import EmbeddingData
from utils.embedding_graph import traverseVNF
from algorithms.hybrid.utils.demand_predictions import DemandPredictions
from algorithms.models.embedding import DecodedIndividual, LinkData
from algorithms.hybrid.constants.surrogate import BRANCH
from algorithms.hybrid.models.traffic import TimeSFCRequests
from algorithms.hybrid.utils.scorer import Scorer
from algorithms.utils.graphs import getVNFsFromFGRs, parseNodes
from shared.models.embedding_graph import VNF, EmbeddingGraph, ForwardingLink
from shared.models.topology import Link, Topology
from dijkstar import Graph, find_path
from shared.models.traffic_design import TrafficDesign
from constants.topology import SERVER, SFCC
from models.calibrate import ResourceDemand
from utils.data import getAvailableCPUAndMemory
from utils.traffic_design import calculateTrafficDuration, getTrafficDesignRate
from utils.tui import TUI

REQUEST_SIZE: float = 0.05  # in Mbps

class MakGAUtils:
    """
    This class contains utility functions for the MAK-GA algorithm.
    """

    _demandPredictions: DemandPredictions = DemandPredictions()
    _topology = None
    _trafficDesign = None
    _fgrs = None

    def __init__(
        self,
        topology: Topology,
        trafficDesign: TrafficDesign,
        fgrs: list[EmbeddingGraph],
    ) -> None:
        self._topology = topology
        self._trafficDesign = trafficDesign
        self._fgrs = fgrs

    def generateRandomIndividual(
        self,
        container: list,
    ) -> list[int]:
        """
        Generates a random individual for the genetic algorithm.

        Parameters:
            container (list): The container to hold the individual.
        Returns:
            list[int]: A list of integers representing the random individual.
        """

        individual: list[int] = container()
        noOfVNFs: int = len(getVNFsFromFGRs(self._fgrs))
        noOfHosts: int = len(self._topology["hosts"])
        isValid: bool = False

        while not isValid:
            for _i in range(noOfVNFs):
                host: int = random.randint(1, noOfHosts)
                if random.random() < 0.1:
                    host = 0
                individual.append(host)

            decodedIndividual = self.decodePop([individual])[0]
            data: TimeSFCRequests = self._generateTrafficData(
                decodedIndividual[1], isMax=True
            )
            MakGAUtils._demandPredictions.cacheResourceDemands(decodedIndividual[1], data)
            # isValid = not isHostConstraintViolated(
            #     decodedIndividual
            # ) and not isLinkConstraintViolated(decodedIndividual)
            isValid = True

        return individual

    def _convertIndividualToEmbeddingGraphs(self, individual: list[int]) -> tuple[
        list[EmbeddingGraph],
        EmbeddingData,
        LinkData,
        dict[str, list[tuple[str, int, int]]],
    ]:
        """
        Converts a list of integers representing an individual into a list of EmbeddingGraph objects.

        Parameters:
            individual (list[int]): A list of integers where each integer represents the index of a host.

        Returns:
            tuple[list[EmbeddingGraph], EmbeddingData, LinkData, dict[str, list[tuple[str, int]]]]:
                A tuple containing:
                - A list of EmbeddingGraph objects.
                - An EmbeddingData object containing the embedding data.
                - A LinkData object containing the link data.
                - A dictionary mapping SFC IDs to lists of tuples containing VNF IDs, their instance, and their depths.
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

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                nodes[fgr["sfcID"]].append(SERVER)

                return

            hostIndex: int = copiedIndividual.pop(0)

            if hostIndex == 0:
                embeddingNotFound[0] = True

                return

            hostID: str = self._topology["hosts"][hostIndex - 1]["id"]
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
                vnfData[fgr["sfcID"]] = [(vnf["vnf"]["id"], vnfInstance, depth)]
            else:
                vnfData[fgr["sfcID"]].append((vnf["vnf"]["id"], vnfInstance, depth))

            if vnf["host"]["id"] in embeddingData:
                if fgr["sfcID"] in embeddingData[vnf["host"]["id"]]:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]].append(
                        [vnf["vnf"]["id"], vnfInstance, depth]
                    )
                else:
                    embeddingData[vnf["host"]["id"]][fgr["sfcID"]] = [
                        [vnf["vnf"]["id"], vnfInstance, depth]
                    ]
            else:
                embeddingData[vnf["host"]["id"]] = {
                    fgr["sfcID"]: [[vnf["vnf"]["id"], vnfInstance, depth]]
                }

        for index, fgr in enumerate(self._fgrs):
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

                for link in self._topology["links"]:
                    graph.add_edge(
                        link["source"],
                        link["destination"],
                        (
                            link["bandwidth"]
                            if "bandwidth" in link and link["bandwidth"] is not None
                            else 1
                        ),
                    )
                    graph.add_edge(
                        link["destination"],
                        link["source"],
                        (
                            link["bandwidth"]
                            if "bandwidth" in link and link["bandwidth"] is not None
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
                                for topoLink in self._topology["links"]
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
        )

    def decodePop(self, pop: list[list[int]]) -> list[
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

        Returns:
            list[tuple[int, list[EmbeddingGraph], EmbeddingData, LinkData, float, dict[str, list[tuple[str, int]]]]]:A list containing a tuple that consists of the index, embedding graphs, embedding data, link data, acceptance ratio, and VNF data for each individual.
        """

        decodedPop: list[DecodedIndividual] = []
        copiedFGRs: list[EmbeddingGraph] = copy.deepcopy(self._fgrs)
        for index, individual in enumerate(pop):
            egs, embeddingData, linkData, vnfData = (
                self._convertIndividualToEmbeddingGraphs(individual)
            )

            acceptanceRatio: float = len(egs) / len(copiedFGRs) if copiedFGRs else 0.0
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

    def _generateTrafficData(
        self,
        egs: list[EmbeddingGraph],
        isMax: bool = False,
    ) -> TimeSFCRequests:
        """
        Generates traffic data from a traffic design.

        Parameters:
            trafficDesign (TrafficDesign): A traffic design object.
            egs (list[EmbeddingGraph]): A list of EmbeddingGraph objects.
            isMax (bool): If True, returns the maximum requests per second; otherwise, returns the average.

        Returns:
            TimeSFCRequests: A TimeSFCRequests object containing the traffic data.
        """

        data: TimeSFCRequests = []
        duration: int = calculateTrafficDuration(self._trafficDesign)
        designRate: list[float] = getTrafficDesignRate(
            self._trafficDesign, [1] * duration
        )
        maxRate: float = max(designRate)
        if isMax:
            data = [{eg["sfcID"]: maxRate for eg in egs}]
        else:
            data = [{eg["sfcID"]: rate for eg in egs} for rate in designRate]
        return data

    def _isVNFInHost(
        self,
        sfcID: str,
        vnfID: str,
        instance: int,
        hostID: str,
        embeddingData: EmbeddingData,
    ) -> bool:
        """
        Checks if a VNF is embedded in a specific host.

        Parameters:
            sfcID (str): The ID of the SFC.
            vnfID (str): The ID of the VNF.
            instance (int): The instance number of the VNF.
            hostID (str): The ID of the host.
            embeddingData (EmbeddingData): The embedding data containing host information.

        Returns:
            bool: True if the VNF is in the host, False otherwise.
        """

        return (
            hostID in embeddingData
            and sfcID in embeddingData[hostID]
            and any(
                vnf[0] == vnfID and vnf[1] == instance
                for vnf in embeddingData[hostID][sfcID]
            )
        )

    def _getProcessingDelay(
        self, data: dict[str, float], decodedIndividual: DecodedIndividual
    ) -> float:
        """
        Calculates the processing delay for a decoded individual based on the topology.

        Parameters:
            data (dict[str, float]): The traffic data containing requests for each SFC.
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total CPU cost for the individual.
        """

        egs: list[EmbeddingGraph] = decodedIndividual[1]
        cpuCost: float = 0.0
        for eg in egs:
            for host in self._topology["hosts"]:
                for _ in range(len(decodedIndividual[5][eg["sfcID"]])):
                    serverCPU, _ = getAvailableCPUAndMemory()
                    cpuAvailable: float = (
                        host["cpu"] if host["cpu"] is not None else serverCPU
                    )
                    totalVNFDemand: float = 0.0

                    for vnf, instance, depth in decodedIndividual[5][eg["sfcID"]]:
                        vnfDemand: float = 0.0
                        divisor: int = 2 ** (depth - 1)
                        if self._isVNFInHost(
                            eg["sfcID"], vnf, instance, host["id"], decodedIndividual[2]
                        ):
                            vnfDemand = MakGAUtils._demandPredictions.getDemand(
                                vnf,
                                (
                                    (data[eg["sfcID"]] / divisor)
                                    if eg["sfcID"] in data
                                    else 0
                                ),
                            )["cpu"]

                        totalVNFDemand += vnfDemand

                    cpuCost += 1 / (cpuAvailable - totalVNFDemand)

        return cpuCost

    def _isVirtualLinkInPhysicalLink(
        self, topoLink: Link, egLink: ForwardingLink
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
        for src, dest in zip(links[::1], links[1::1]):
            if (src == topoLink["source"] and dest == topoLink["destination"]) or (
                src == topoLink["destination"] and dest == topoLink["source"]
            ):
                return True

        return False

    def _getPropagationDelay(self, decodedIndividual: DecodedIndividual) -> float:
        """
        Calculates the propagation delay for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total propagation delay for the individual.
        """

        egs: list[EmbeddingGraph] = decodedIndividual[1]

        propagationDelay: float = 0.0
        for eg in egs:
            for topoLink in self._topology["links"]:
                for link in eg["links"]:
                    if self._isVirtualLinkInPhysicalLink(topoLink, link):
                        propagationDelay += (
                            topoLink["delay"] if "delay" in topoLink else 0
                        )

        return propagationDelay

    def _getQueueDelay(
        self,
        data: dict[str, float],
        decodedIndividual: DecodedIndividual,
    ) -> float:
        """
        Calculates the queue delay for a decoded individual based on the topology.

        Parameters:
            data (dict[str, float]): The traffic data containing requests for each SFC.
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total queue delay for the individual.
        """

        egs: list[EmbeddingGraph] = decodedIndividual[1]
        linkData: LinkData = decodedIndividual[3]
        queueDelay: float = 0.0

        for eg in egs:
            for physicLink, physicLinkData in linkData.items():
                if eg["sfcID"] not in physicLinkData:
                    continue

                src: str = physicLink.split("-")[0]
                dest: str = physicLink.split("-")[1]
                topoLink: Link = [
                    link
                    for link in self._topology["links"]
                    if (link["source"] == src and link["destination"] == dest)
                    or (link["source"] == dest and link["destination"] == src)
                ][0]
                for _ in range(len(eg["links"])):
                    totalLinkDemand: float = 0.0

                    # for link in eg["links"]:
                    #     linkDemand: float = 0.0
                    #     if isVirtualLinkInPhysicalLink(topoLink, link):
                    #         linkDemand = (
                    #             (data[eg["sfcID"]] * physicLinkData[eg["sfcID"]][0])
                    #             if eg["sfcID"] in data
                    #             else 0
                    #         )

                    #     totalLinkDemand += linkDemand

                    totalLinkDemand = (
                        data[eg["sfcID"]] * physicLinkData[eg["sfcID"]][0]
                        if eg["sfcID"] in data
                        else 0
                    )

                    totalLinkDemandSize: float = totalLinkDemand * REQUEST_SIZE
                    queueDelay += 1 / (topoLink["bandwidth"] - totalLinkDemandSize)

        return queueDelay

    def getVirtualisationDelay(self, decodedIndividual: DecodedIndividual) -> float:
        """
        Calculates the virtualisation delay for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total CPU cost for the individual.
        """

        egs: list[EmbeddingGraph] = decodedIndividual[1]
        hostVirtualisationDelay: float = 1.0

        virtualisationDelay: float = 0.0
        for eg in egs:
            for host in self._topology["hosts"]:
                for vnf, instance, _depth in decodedIndividual[5][eg["sfcID"]]:
                    if self._isVNFInHost(
                        eg["sfcID"], vnf, instance, host["id"], decodedIndividual[2]
                    ):
                        virtualisationDelay += hostVirtualisationDelay

        return virtualisationDelay

    def _isHostConstraintViolated(self, decodedIndividual: DecodedIndividual) -> bool:
        """
        Checks if the host constraints are violated for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total CPU cost for the individual.
        """

        egs: list[EmbeddingGraph] = decodedIndividual[1]
        data: dict[str, float] = self._generateTrafficData(egs, isMax=True)[0]
        embeddingData: EmbeddingData = copy.deepcopy(decodedIndividual[2])
        for host, sfc in embeddingData.items():
            for sfcID, vnfs in sfc.items():
                embeddingData[host][sfcID] = []
                for vnf in vnfs:
                    embeddingData[host][sfcID].append(
                        (
                            vnf[0],
                            vnf[2],
                        )
                    )
        scores: dict[str, ResourceDemand] = Scorer.getHostScores(
            data, self._topology, embeddingData, MakGAUtils._demandPredictions
        )[1]

        if len(scores) == 0:
            return False

        maxCPU: float = max(scores.values(), key=lambda score: score["cpu"])["cpu"]
        maxMemory: float = max(scores.values(), key=lambda score: score["memory"])[
            "memory"
        ]

        return maxCPU > 1.0 or maxMemory > 1.0

    def _isLinkConstraintViolated(self, decodedIndividual: DecodedIndividual) -> bool:
        """
        Checks if the link constraints are violated for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            bool: True if the link constraints are violated, False otherwise.
        """
        egs: list[EmbeddingGraph] = decodedIndividual[1]
        linkData: LinkData = decodedIndividual[3]
        data: TimeSFCRequests = self._generateTrafficData(egs, isMax=True)[0]

        linkRequestData: dict[str, float] = {}
        for eg in egs:
            checkedLinks: set[str] = set()
            for egLink in eg["links"]:
                links: "list[str]" = [egLink["source"]["id"]]
                links.extend(egLink["links"])
                links.append(egLink["destination"]["id"])

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    if f"{source}-{destination}" in linkData:
                        if f"{source}-{destination}" in checkedLinks:
                            continue
                        checkedLinks.add(f"{source}-{destination}")
                        for key, pathData in linkData[
                            f"{source}-{destination}"
                        ].items():
                            reqps: float = data[key] if key in data else 0
                            if f"{source}-{destination}" in linkRequestData:
                                linkRequestData[f"{source}-{destination}"] += (
                                    pathData[0] * reqps
                                )
                            else:
                                linkRequestData[f"{source}-{destination}"] = (
                                    pathData[0] * reqps
                                )
                    elif f"{destination}-{source}" in linkData:
                        if f"{destination}-{source}" in checkedLinks:
                            continue
                        checkedLinks.add(f"{destination}-{source}")
                        for key, pathData in linkData[
                            f"{destination}-{source}"
                        ].items():
                            reqps: float = data[key] if key in data else 0
                            if f"{destination}-{source}" in linkRequestData:
                                linkRequestData[f"{destination}-{source}"] += (
                                    pathData[0] * reqps
                                )
                            else:
                                linkRequestData[f"{destination}-{source}"] = (
                                    pathData[0] * reqps
                                )

        for key, linkRequests in linkRequestData.items():
            source, destination = key.split("-")
            bandwidth: float = [
                link["bandwidth"]
                for link in self._topology["links"]
                if (link["source"] == source and link["destination"] == destination)
                or (link["source"] == destination and link["destination"] == source)
            ][0]

            requestsSize: float = linkRequests * REQUEST_SIZE
            if requestsSize > bandwidth:
                return True

        return False

    def getTotalDelay(self, decodedIndividual: DecodedIndividual) -> float:
        """
        Calculates the total delay for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            float: The total delay for the individual.
        """

        propagationDelay: float = self._getPropagationDelay(
            decodedIndividual
        )
        virtualisationDelay: float = self.getVirtualisationDelay(
            decodedIndividual
        )
        duration: int = calculateTrafficDuration(self._trafficDesign)
        trafficRate: "list[float]" = getTrafficDesignRate(
            self._trafficDesign, [1] * duration
        )
        avgData: TimeSFCRequests = {
            eg["sfcID"]: np.mean(trafficRate) for eg in decodedIndividual[1]
        }
        processingDelay: float = self._getProcessingDelay(avgData, decodedIndividual)
        queueDelay: float = self._getQueueDelay(
            avgData, decodedIndividual
        )

        return processingDelay + queueDelay + propagationDelay + virtualisationDelay

    def cacheDemand(self, pop: list[DecodedIndividual]) -> None:
        """
        Caches the resource demands for each embedding graph based on the provided data.

        Parameters:
            pop (list[DecodedIndividual]): List of decoded individuals.

        Returns:
            None
        """

        egs: list[EmbeddingGraph] = []
        data: TimeSFCRequests = []
        for ind in pop:
            if ind[4] == 0:
                continue
            egs.extend(ind[1])

        egs = copy.deepcopy(egs)

        for i, eg in enumerate(egs):
            eg["sfcID"] = f"{eg['sfcID']}_{i}" if "sfcID" in eg else f"sfc{i}"

        trafficData: TimeSFCRequests = self._generateTrafficData(egs)
        maxData: TimeSFCRequests = self._generateTrafficData(egs, isMax=True)
        data.extend(trafficData)
        data.extend(maxData)

        MakGAUtils._demandPredictions.cacheResourceDemands(egs, data)
