"""
Defines the CPU, memory and link scorer.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
from threading import Thread
import timeit
from typing import Any, Tuple
import polars as pl
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Host, Topology
from algorithms.surrogacy.utils.demand_predictions import DemandPredictions
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF


class Scorer:
    """
    This defines the scorer that computes different scoring metrics used by the surrogate model.
    """

    def getHostScores(
        self,
        data: pl.DataFrame,
        topology: Topology,
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        demandPredictor: DemandPredictions,
    ) -> "dict[str, ResourceDemand]":
        """
        Gets the host scores.

        Parameters:
            data (pl.DataFrame): the data.
            topology (Topology): the topology.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
            demandPredictor (Predictions): the demand predictor.

        Returns:
            dict[str, ResourceDemand]: the host scores.
        """

        hostResourceData: "dict[str, ResourceDemand]" = {}
        for host, sfcs in embeddingData.items():
            otherCPU: float = 0
            otherMemory: float = 0

            for sfc, vnfs in sfcs.items():
                for vnf, depth in vnfs:
                    divisor: int = 2 ** (depth - 1)
                    sfcData: pl.DataFrame = data.filter(pl.col("sfc") == sfc)
                    effectiveReqps: float = (
                        (pl.Series(sfcData.select("reqps"))[0] / divisor)
                        if sfcData.height > 0
                        else 0
                    )
                    demands: ResourceDemand = demandPredictor.getDemand(
                        vnf, effectiveReqps
                    )
                    vnfCPU: float = demands["cpu"]
                    vnfMemory: float = demands["memory"]
                    otherCPU += vnfCPU
                    otherMemory += vnfMemory

            hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)

        for host, data in hostResourceData.items():
            topoHost: Host = [h for h in topology["hosts"] if h["id"] == host][0]
            hostCPU: float = topoHost["cpu"]
            hostMemory: float = topoHost["memory"]
            data["cpu"] = data["cpu"] / hostCPU if hostCPU is not None else 0
            data["memory"] = (
                data["memory"] / hostMemory if hostMemory is not None else 0
            )

        return hostResourceData

    def getLinkScores(
        self,
        data: pl.DataFrame,
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        linkData: "dict[str, dict[str, float]]",
    ) -> "dict[str, float]":
        """
        Gets the link scores.

        Parameters:
            data pl.DataFrame: the data.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            linkData (dict[str, dict[str, float]]): the link data.

        Returns:
            dict[str, float]: the link scores.
        """

        linkScores: "dict[str, float]" = {}
        for eg in egs:
            totalLinkScore: float = 0.0
            for egLink in eg["links"]:
                links: "list[str]" = [egLink["source"]["id"]]
                links.extend(egLink["links"])
                links.append(egLink["destination"]["id"])
                divisor: int = egLink["divisor"]
                sfcData: pl.DataFrame = data.filter(pl.col("sfc") == eg["sfcID"])
                reqps: float = (
                    (pl.Series(sfcData.select("reqps"))[0] / divisor) if sfcData.height > 0 else 0
                )

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    totalRequests: int = 0

                    if f"{source}-{destination}" in linkData:
                        for key, factor in linkData[f"{source}-{destination}"].items():
                            sfcData: pl.DataFrame = data.filter(pl.col("sfc") == key)
                            totalRequests += factor * (
                                pl.Series(sfcData.select("reqps"))[0] if sfcData.height > 0 else 0
                            )
                    elif f"{destination}-{source}" in linkData:
                        for key, factor in linkData[f"{destination}-{source}"].items():
                            sfcData: pl.DataFrame = data.filter(pl.col("sfc") == key)
                            totalRequests += factor * (
                                pl.Series(sfcData.select("reqps"))[0] if sfcData.height > 0 else 0
                            )

                    bandwidth: float = [
                        link["bandwidth"]
                        for link in topology["links"]
                        if (
                            link["source"] == source
                            and link["destination"] == destination
                        )
                        or (
                            link["source"] == destination
                            and link["destination"] == source
                        )
                    ][0]

                    linkScore: float = self._getLinkScore(
                        reqps, totalRequests, bandwidth
                    )

                    totalLinkScore += linkScore

            linkScores[eg["sfcID"]] = totalLinkScore

        return linkScores

    def getSFCScores(
        self,
        data: pl.DataFrame,
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
            pl.DataFrame: the SFC scores.
        """

        outputData: pl.DataFrame = None

        def addHostLinkScoresForEachTimeSlot(
            timeData: pl.DataFrame
        ) -> pl.DataFrame:
            """
            Adds the host and link scores for each time slot.

            Parameters:
                timeData (pl.DataFrame): the time data.

            Returns:
                pl.DataFrame: A new DataFrame with CPU, memory and link scores for the specific time slot.
            """

            hostScores: "dict[str, ResourceDemand]" = self.getHostScores(
                timeData, topology, embeddingData, demandPredictor
            )
            linkScores: "dict[str, float]" = self.getLinkScores(
                timeData, topology, egs, linkData
            )

            timeSlotData: pl.DataFrame = None

            def addHostLinkScoresForSFC(
                sfcData: dict[str, Any],
                hostScores: "dict[str, ResourceDemand]",
                linkScores: "dict[str, float]",
            ) -> pl.DataFrame:
                """
                Adds the host and link scores for a specific SFC.

                Parameters:
                    sfcData (pl.Series): the SFC data.
                    hostScores (dict[str, ResourceDemand]): the host scores.
                    linkScores (dict[str, float]): the link scores.

                Returns:
                    pl.DataFrame: A new DataFrame with CPU, memory and link scores for the specific SFC in teh specific time slot.
                """

                totalCPUScore: float = 0
                totalMemoryScore: float = 0
                eg: EmbeddingGraph = [
                    graph for graph in egs if graph["sfcID"] == sfcData["sfc"]
                ][0]
                hosts: "dict[str, ResourceDemand]" = {}
                totalHostCPU: float = 0
                totalHostMemory: float = 0

                def parseVNF(vnf: VNF, depth: int) -> None:
                    """
                    Parses a VNF.

                    Parameters:
                        vnf (VNF): the VNF.
                        depth (int): the depth.
                    """

                    nonlocal totalCPUScore
                    nonlocal totalMemoryScore
                    nonlocal totalHostCPU
                    nonlocal totalHostMemory

                    divisor: int = 2 ** (depth - 1)
                    reqps: float = sfcData["reqps"] / divisor
                    demands: ResourceDemand = demandPredictor.getDemand(
                        vnf["vnf"]["id"], reqps
                    )

                    host: Host = [
                        host
                        for host in topology["hosts"]
                        if host["id"] == vnf["host"]["id"]
                    ][0]
                    hostCPU: float = host["cpu"]
                    hostMemory: float = host["memory"]

                    vnfCPU: float = demands["cpu"]
                    vnfMemory: float = demands["memory"]

                    totalCPUScore += self._getScore(vnfCPU, hostCPU)
                    totalMemoryScore += self._getScore(vnfMemory, hostMemory)

                    totalHostCPU += hostScores[vnf["host"]["id"]]["cpu"]
                    totalHostMemory += hostScores[vnf["host"]["id"]]["memory"]

                    if vnf["host"]["id"] not in hosts:
                        hosts.update({vnf["host"]["id"]: hostScores[vnf["host"]["id"]]})

                traverseVNF(eg["vnfs"], parseVNF, shouldParseTerminal=False)

                maxCPU: float = (
                    max([host["cpu"] for host in hosts.values()])
                    if len(hosts) > 0
                    else 0.0
                )
                maxMemory: float = (
                    max([host["memory"] for host in hosts.values()])
                    if len(hosts) > 0
                    else 0.0
                )
                link: float = (
                    linkScores[sfcData["sfc"]]
                    if linkScores is not None and sfcData["sfc"] in linkScores
                    else 0.0
                )

                newData: pl.DataFrame = pl.DataFrame(
                    {
                        "generation": [sfcData["generation"]],
                        "individual": [sfcData["individual"]],
                        "time": [sfcData["time"]],
                        "sfc": [sfcData["sfc"]],
                        "real_reqps": [sfcData["real_reqps"]],
                        "latency": [sfcData["latency"]],
                        "ar": [sfcData["ar"]],
                        "reqps": [sfcData["reqps"]],
                        "max_cpu": [maxCPU],
                        "max_memory": [maxMemory],
                        "link": [link],
                    }
                )

                return newData

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        addHostLinkScoresForSFC, sfcData, hostScores, linkScores
                    )
                    for sfcData in timeData.iter_rows(named=True)
                ]
                for future in futures:
                    sfc = future.result()
                    if timeSlotData is None:
                        timeSlotData = sfc
                    else:
                        timeSlotData = pl.concat([timeSlotData, sfc])
            # for sfcData in timeData.iter_rows(named=True):
            #     sfc = addHostLinkScoresForSFC(sfcData, hostScores, linkScores)
            #     if timeSlotData is None:
            #         timeSlotData = sfc
            #     else:
            #         timeSlotData = pl.concat([timeSlotData, sfc])

            return timeSlotData
        start = timeit.default_timer()

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(addHostLinkScoresForEachTimeSlot, groupedData)
                for _time, groupedData in data.group_by("time")
            ]

            for future in futures:
                time = future.result()
                if outputData is None:
                    outputData = time
                else:
                    outputData = pl.concat([outputData, time])
        # for _time, groupedData in data.group_by("time"):
        #     time = addHostLinkScoresForEachTimeSlot(groupedData)
        #     if outputData is None:
        #         outputData = time
        #     else:
        #         outputData = pl.concat([outputData, time])
        end = timeit.default_timer()
        print(f"Time taken to compute SFC scores: {end - start} seconds")

        return outputData

    def _getScore(self, demand: float, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            demand (float): the demand.
            resource (float): the resource.

        Returns:
            float: the score.
        """

        return (demand / resource) if resource is not None and demand is not None else 0

    def _getLinkScore(self, demand: float, totalDemand: int, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            demand (float): the demand.
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
