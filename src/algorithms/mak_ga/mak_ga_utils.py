"""
This defines the util class used by the GA algorithm developed by Mohammad Ali Khoshkholghi.
"""

from concurrent.futures import ProcessPoolExecutor
import copy
import random
from typing import cast
from algorithms.mak_ga.gaha_individual import convertIndividualToEmbeddingGraphs
import numpy as np
from shared.models.sfc_request import SFCRequest
from algorithms.hybrid.models.individuals import Individual
from algorithms.mak_ga.models.embedding import EmbeddingData
from algorithms.hybrid.utils.demand_predictions import DemandPredictions
from algorithms.models.embedding import DecodedIndividual, LinkData
from algorithms.hybrid.models.traffic import TimeSFCRequests
from algorithms.hybrid.utils.scorer import Scorer
from algorithms.utils.graphs import convertSFCRsToEGs
from shared.models.embedding_graph import EmbeddingGraph, ForwardingLink
from shared.models.topology import Link, Topology
from shared.models.traffic_design import TrafficDesign
from models.calibrate import ResourceDemand
from utils.data import getAvailableCPUAndMemory
from utils.traffic_design import calculateTrafficDuration, getTrafficDesignRate

REQUEST_SIZE: float = 0.05  # in Mbps

class MakGAUtils:
    """
    This class contains utility functions for the MAK-GA algorithm.
    """

    _demandPredictions: DemandPredictions = DemandPredictions()

    def __init__(
        self,
        topology: Topology,
        trafficDesign: TrafficDesign,
        sfcrs: list[SFCRequest],
    ) -> None:
        self._topology = topology
        self._trafficDesign = trafficDesign
        self._sfcrs = sfcrs
        self._fgrs = convertSFCRsToEGs(sfcrs)


    def mutate(self, individual: Individual, indpb: float) -> Individual:
        """
        Mutates an individual by randomly changing its genes based on a given probability.

        Parameters:
            individual (Individual): The individual to mutate.
            indpb (float): The probability of mutating each gene.

        Returns:
            Individual: The mutated individual.
        """

        noOfHosts: int = len(self._topology["hosts"])

        for i in range(len(individual)):
            if random.random() < indpb:
                host: int = random.randint(1, noOfHosts)
                individual[i] = host

        return individual


    def decodePop(self, pop: list[Individual], ignoreVNFInstances: bool = False) -> list[DecodedIndividual]:
        """
        Decodes a population of individuals into EmbeddingGraph objects and calculates the total cost.

        Parameters:
            pop (list[Individual]): A list of individuals, where each individual is an Individual object.
            ignoreVNFInstances (bool): If True, ignores VNF instances when decoding.

        Returns:
            list[DecodedIndividual]: A list containing a tuple that consists of the index, embedding graphs, embedding data, link data, acceptance ratio, and VNF data for each individual.
        """

        decodedPop: list[DecodedIndividual] = []
        copiedFGRs: list[SFCRequest] = copy.deepcopy(self._fgrs)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    convertIndividualToEmbeddingGraphs,
                    individual,
                    self._topology,
                    self._fgrs,
                    index,
                    ignoreVNFInstances
                )
                for index, individual in enumerate(pop)
            ]

            for future in futures:
                egs, embeddingData, linkData, vnfData, index = future.result()
                acceptanceRatio: float = len(egs) / len(copiedFGRs) if copiedFGRs else 0.0
                decodedPop.append(cast(DecodedIndividual, (index, egs, embeddingData, linkData, acceptanceRatio, pop[index].id)))

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

    def getTotalDelay(self, decodedIndividual: DecodedIndividual) -> tuple[int, float, float]:
        """
        Calculates the total delay for a decoded individual based on the topology.

        Parameters:
            decodedIndividual (DecodedIndividual): The decoded individual containing embedding data.

        Returns:
            tuple[int, float, float]: The index, acceptance ratio and the latency.
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
            eg["sfcID"]: np.median(trafficRate) for eg in decodedIndividual[1]
        }
        processingDelay: float = self._getProcessingDelay(avgData, decodedIndividual)
        queueDelay: float = self._getQueueDelay(
            avgData, decodedIndividual
        )

        return decodedIndividual[0], decodedIndividual[4], processingDelay + queueDelay + propagationDelay + virtualisationDelay

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

    def rejectVNF(self, individual: Individual, rejectionRate: float) -> Individual:
        """
        Determines whether to reject a VNF based on the rejection rate.

        Parameters:
            individual (Individual): The individual to evaluate.
            rejectionRate (float): The probability of rejection.

        Returns:
            Individual: The individual after potential rejection.
        """

        for i in range(len(individual)):
            if random.random() < rejectionRate:
                individual[i] = 0

        return individual
