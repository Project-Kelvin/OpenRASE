"""
Defines the CPU, memory and link scorer.
"""

from typing import Tuple
import numpy as np
import polars as pl
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Host, Link, Topology
from algorithms.models.embedding import EmbeddingData, LinkData
from algorithms.hybrid.models.traffic import TimeSFCRequests
from algorithms.hybrid.utils.demand_predictions import DemandPredictions
from models.calibrate import ResourceDemand
from utils.data import getAvailableCPUAndMemory
from utils.embedding_graph import traverseVNF


class Scorer:
    """
    This defines the scorer that computes different scoring metrics used by the surrogate model.
    """

    @staticmethod
    def getHostScores(
        data: dict[str, float],
        topology: Topology,
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        demandPredictor: DemandPredictions,
    ) -> "dict[str, ResourceDemand]":
        """
        Gets the host scores.

        Parameters:
            data (dict[str, float]): dictionary containing SFCs and the reqs/s for the given time slot.
            topology (Topology): the topology.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
            demandPredictor (Predictions): the demand predictor.

        Returns:
            dict[str, ResourceDemand]: the host scores.
        """

        hostResourceData: "dict[str, ResourceDemand]" = {}
        serverCPU, serverMemory = getAvailableCPUAndMemory()
        for host, sfcs in embeddingData.items():
            topoHost: Host = [h for h in topology["hosts"] if h["id"] == host][0]
            otherCPU: float = 0
            otherMemory: float = 0

            for sfc, vnfs in sfcs.items():
                for vnf, depth in vnfs:
                    divisor: int = 2 ** (depth - 1)
                    effectiveReqps: float = (data[sfc] / divisor) if sfc in data else 0
                    demands: ResourceDemand = demandPredictor.getDemand(
                        vnf, effectiveReqps
                    )
                    vnfCPU: float = demands["cpu"]
                    vnfMemory: float = demands["memory"]
                    otherCPU += vnfCPU
                    otherMemory += vnfMemory

            hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)
            hostCPU: float = topoHost["cpu"]
            hostMemory: float = topoHost["memory"]
            hostResourceData[host]["cpu"] = Scorer._getScore(
                hostResourceData[host]["cpu"],
                hostCPU if hostCPU is not None else serverCPU,
            )
            hostResourceData[host]["memory"] = Scorer._getScore(
                hostResourceData[host]["memory"],
                hostMemory if hostMemory is not None else serverMemory
            )

        return hostResourceData

    @staticmethod
    def getLinkScores(
        data: dict[str, float],
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        linkData: "dict[str, dict[str, tuple[float, float, int ]]]",
    ) -> "dict[str, float]":
        """
        Gets the link scores.

        Parameters:
            data (dict[str, float]): dictionary containing SFCs and the reqs/s for the given time slot.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            linkData (dict[str, dict[str, float]]): the link data.

        Returns:
            dict[str, tuple[float, float, int ]]: the link scores.
        """

        linkScoresData: "dict[str, tuple[float, float, int ]]" = {}
        for eg in egs:
            totalLinkScore: float = 0.0
            linkScores: list[float] = []
            totalDelay: int = 0
            checkedLinks: set[str] = set()
            for egLink in eg["links"]:
                links: "list[str]" = [egLink["source"]["id"]]
                links.extend(egLink["links"])
                links.append(egLink["destination"]["id"])

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    totalRequests: int = 0

                    link: Link = [
                        topoLink
                        for topoLink in topology["links"]
                        if (
                            topoLink["source"] == source
                            and topoLink["destination"] == destination
                        )
                        or (
                            topoLink["source"] == destination
                            and topoLink["destination"] == source
                        )
                    ][0]

                    if f"{source}-{destination}" in linkData:
                        if f"{source}-{destination}" in checkedLinks:
                            continue
                        checkedLinks.add(f"{source}-{destination}")
                        for key, pathData in linkData[f"{source}-{destination}"].items():
                            reqps: float = (
                                data[key] if key in data else 0
                            )
                            totalRequests += pathData[0] * reqps

                            if key == eg["sfcID"]:
                                totalDelay += pathData[1]
                    elif f"{destination}-{source}" in linkData:
                        if f"{destination}-{source}" in checkedLinks:
                            continue
                        checkedLinks.add(f"{destination}-{source}")
                        for key, pathData in linkData[f"{destination}-{source}"].items():
                            reqps: float = (
                                data[key] if key in data else 0
                            )
                            totalRequests += pathData[0] * reqps

                            if key == eg["sfcID"]:
                                totalDelay += pathData[1]

                    bandwidth: float = link["bandwidth"] if "bandwidth" in link else 1.0
                    linkScore: float = Scorer._getLinkScore(totalRequests, bandwidth)
                    totalLinkScore += linkScore
                    linkScores.append(linkScore)

            linkScoresData[eg["sfcID"]] = (totalLinkScore, max(linkScores), (2 * totalDelay))

        return linkScoresData

    @staticmethod
    def getHostLinkScoresForEachSFC(
        time: int,
        sfc: str,
        sfcReqps: float,
        hostScores: "dict[str, ResourceDemand]",
        linkScores: "dict[str, float]",
        egs: "list[EmbeddingGraph]",
    ) -> np.array:
        """
        Adds the host and link scores for a specific SFC.

        Parameters:
            sfcData (pl.Series): the SFC data.
            hostScores (dict[str, ResourceDemand]): the host scores.
            linkScores (dict[str, float]): the link scores.
            egs (list[EmbeddingGraph]): the Embedding Graphs.

        Returns:
            np.array : A numpy array with CPU, memory and link scores for the specific SFC in teh specific time slot.
        """

        eg: EmbeddingGraph = [graph for graph in egs if graph["sfcID"] == sfc][0]
        hosts: "dict[str, ResourceDemand]" = {}

        def parseVNF(vnf: VNF, depth: int) -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                depth (int): the depth.
            """

            if vnf["host"]["id"] not in hosts:
                hosts.update({vnf["host"]["id"]: hostScores[vnf["host"]["id"]]})

        traverseVNF(eg["vnfs"], parseVNF, shouldParseTerminal=False)

        maxCPU: float = (
            max([host["cpu"] for host in hosts.values()]) if len(hosts) > 0 else 0.0
        )

        maxMemory: float = (
            max([host["memory"] for host in hosts.values()]) if len(hosts) > 0 else 0.0
        )

        totalLinkScore: float = (
            linkScores[sfc][0] if linkScores is not None and sfc in linkScores else 0.0
        )
        maxLinkScore: float = (
            linkScores[sfc][1] if linkScores is not None and sfc in linkScores else 0.0
        )
        totalDelay: int = (
            linkScores[sfc][2] if linkScores is not None and sfc in linkScores else 0
        )

        dt = np.dtype(
            [
                ("time", np.int32),
                ("sfc", "U", 20),
                ("reqps", np.float64),
                ("max_cpu", np.float64),
                ("max_memory", np.float64),
                ("total_link_score", np.float64),
                ("max_link_score", np.float64),
                ("total_delay", np.int32),
            ]
        )
        newData: np.array = np.array(
            [
                (
                    int(time),
                    str(sfc),
                    float(sfcReqps),
                    float(maxCPU),
                    float(maxMemory),
                    float(totalLinkScore),
                    float(maxLinkScore),
                    int(totalDelay)
                )
            ],
            dtype=dt,
        )

        return newData

    @staticmethod
    def getHostLinkScoresForEachTimeSlot(
        time: int,
        timeData: dict[str, float],
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        embeddingData: EmbeddingData,
        linkData: LinkData,
        demandPredictor: DemandPredictions,
    ) -> np.array:
        """
        Adds the host and link scores for each time slot.

        Parameters:
            time (int): the time slot.
            timeData (pl.DataFrame): the time data.

        Returns:
            np.array: A numpy array with CPU, memory and link scores for the specific time slot.
        """

        hostScores: "dict[str, ResourceDemand]" = Scorer.getHostScores(
            timeData, topology, embeddingData, demandPredictor
        )
        linkScores: "dict[str, tuple[float, float, int ]]" = Scorer.getLinkScores(
            timeData, topology, egs, linkData
        )
        hostLinkScores: np.array = None
        for key, value in timeData.items():
            hostLinkScore: np.array = Scorer.getHostLinkScoresForEachSFC(
                time, key, value, hostScores, linkScores, egs
            )
            if hostLinkScores is None:
                hostLinkScores = hostLinkScore
            else:
                hostLinkScores = np.concatenate((hostLinkScores, hostLinkScore))

        return hostLinkScores

    @staticmethod
    def getSFCScores(
        data: TimeSFCRequests,
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        linkData: "dict[str, dict[str, float]]",
        demandPredictor: DemandPredictions,
    ) -> pl.DataFrame:
        """
        Gets the SFC scores.

        Parameters:
            data pl.DataFrame: the data.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
            linkData (dict[str, dict[str, float]]): the link data.
            demandPredictor (Predictions): the demand predictor.

        Returns:
            np.array: the SFC scores (time, SFC ID, reqps, max CPU, max memory, link score).
        """

        outputData: np.array = None
        for time, timeData in enumerate(data):
            timeSlotData: np.array = Scorer.getHostLinkScoresForEachTimeSlot(
                time,
                timeData,
                topology,
                egs,
                embeddingData,
                linkData,
                demandPredictor,
            )
            if outputData is None:
                outputData = timeSlotData
            else:
                outputData = np.concatenate((outputData, timeSlotData))

        return outputData

    @staticmethod
    def _getScore(demand: float, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            demand (float): the demand.
            resource (float): the resource.

        Returns:
            float: the score.
        """

        return (demand / resource) if resource is not None and demand is not None else 0

    @staticmethod
    def _getLinkScore(totalDemand: int, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            totalDemand (int): the total demand.
            resource (float): the resource.

        Returns:
            float: the score.
        """

        return (
            (totalDemand / resource)
            if resource is not None and totalDemand is not None
            else 0
        )
