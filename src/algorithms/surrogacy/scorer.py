"""
Defines the CPU, memory and link scorer.
"""

from typing import Tuple
import pandas as pd
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Host, Topology
from calibrate.calibrate import Calibrate
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF


class Scorer:
    """
    This defines the scorer that computes different scoring metrics used by the surrogate model.
    """

    def __init__(self) -> None:
        """
        Initializes the scorer.
        """

        self._calibrate: Calibrate = Calibrate()

    def getHostScores(
        self,
        data: pd.DataFrame,
        topology: Topology,
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
    ) -> "dict[str, ResourceDemand]":
        """
        Gets the host scores.

        Parameters:
            data (pd.DataFrame): the data.
            topology (Topology): the topology.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.

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
                    effectiveReqps: float = (
                        (data.loc[data["sfc"] == sfc, "reqps"].iloc[0] / divisor)
                        if sfc in data["sfc"].values
                        else 0
                    )
                    demands: ResourceDemand = (
                        self._calibrate.getVNFResourceDemandForReqps(
                            vnf, effectiveReqps
                        )
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
            data["cpu"] = data["cpu"] / hostCPU
            data["memory"] = data["memory"] / hostMemory

        return hostResourceData

    def getLinkScores(
        self,
        data: pd.DataFrame,
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        linkData: "dict[str, dict[str, float]]",
    ) -> "dict[str, float]":
        """
        Gets the link scores.

        Parameters:
            data pd.DataFrame: the data.
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
                reqps: float = (
                    (data.loc[data["sfc"] == eg["sfcID"], "reqps"].iloc[0] / divisor)
                    if eg["sfcID"] in data["sfc"].values
                    else 0
                )

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    totalRequests: int = 0

                    if f"{source}-{destination}" in linkData:
                        for key, factor in linkData[f"{source}-{destination}"].items():
                            totalRequests += factor * (
                                data.loc[data["sfc"] == key, "reqps"].iloc[0]
                                if key in data["sfc"].values
                                else 0
                            )
                    elif f"{destination}-{source}" in linkData:
                        for key, factor in linkData[f"{destination}-{source}"].items():
                            totalRequests += factor * (
                                data.loc[data["sfc"] == key, "reqps"].iloc[0]
                                if key in data["sfc"].values
                                else 0
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

    def cacheData(self, data: pd.DataFrame, egs: "list[EmbeddingGraph]") -> None:
        """
        Caches the data.

        Parameters:
            data (pd.DataFrame): the data.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
        """

        vnfsInEGs: "dict[str, set[str]]" = {}

        def parseEG(vnf: VNF, _depth: int, egID: str) -> None:
            """
            Parses an EG.

            Parameters:
                vnf (VNF): the VNF.
                _depth (int): the depth.
                egID (str): the EG ID.
            """

            nonlocal vnfsInEGs

            if egID not in vnfsInEGs:
                vnfsInEGs[egID] = {vnf["vnf"]["id"]}
            else:
                vnfsInEGs[egID].add(vnf["vnf"]["id"])

        for eg in egs:
            traverseVNF(eg["vnfs"], parseEG, eg["sfcID"], shouldParseTerminal=False)

        dataToCache: "dict[str, list[float]]" = {}

        for _index, sfcData in data.iterrows():
            for vnf in vnfsInEGs[sfcData["sfc"]]:
                if vnf in dataToCache:
                    dataToCache[vnf].append(sfcData["reqps"])
                else:
                    dataToCache[vnf] = [sfcData["reqps"]]

        self._calibrate = Calibrate()
        self._calibrate.predictAndCache(dataToCache)

    def getSFCScores(
        self,
        data: pd.DataFrame,
        topology: Topology,
        egs: "list[EmbeddingGraph]",
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        linkData: "dict[str, dict[str, float]]",
    ) -> pd.DataFrame:
        """
        Gets the SFC scores.

        Parameters:
            data pd.DataFrame: the data.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
            linkData (dict[str, dict[str, float]]): the link data.

        Returns:
            pd.DataFrame: the SFC scores.
        """

        data["max_cpu"] = 0.0
        data["link"] = 0.0
        data = data.rename_axis("rowIndex").reset_index()

        for _time, groupedData in data.groupby("time"):
            hostScores: "dict[str, ResourceDemand]" = self.getHostScores(
                groupedData, topology, embeddingData
            )
            linkScores: "dict[str, float]" = self.getLinkScores(
                groupedData, topology, egs, linkData
            )

            for sfcData in groupedData.itertuples():
                totalCPUScore: float = 0
                totalMemoryScore: float = 0
                eg: EmbeddingGraph = [
                    graph for graph in egs if graph["sfcID"] == sfcData.sfc
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
                    reqps: float = sfcData.reqps / divisor
                    demands: ResourceDemand = (
                        self._calibrate.getVNFResourceDemandForReqps(
                            vnf["vnf"]["id"], reqps
                        )
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

                data.loc[data["rowIndex"] == sfcData.rowIndex, "max_cpu"] = max(
                    [host["cpu"] for host in hosts.values()]
                )
                data.loc[data["rowIndex"] == sfcData.rowIndex, "link"] = linkScores[
                    sfcData.sfc
                ]

        data.drop(columns=["rowIndex"], inplace=True)
        data.to_csv("data.csv")
        return data

    def _getScore(self, demand: float, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            demand (float): the demand.
            resource (float): the resource.

        Returns:
            float: the score.
        """

        return demand / resource

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

        return totalDemand / resource
