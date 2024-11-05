"""
Defines the CPU, memory and link scorer.
"""

from typing import Tuple, Union
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Host, Topology
from calibrate.calibrate import Calibrate
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF
from utils.tui import TUI


class Scorer():
    """
    This defines the scorer that computes different scoring metrics used by the surrogate model.
    """

    def __init__(self) -> None:
        """
        Initializes the scorer.
        """

        self._calibrate: Calibrate = Calibrate()


    def getHostScores(self, reqps: int, topology: Topology, egs: "list[EmbeddingGraph]", embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]" ) -> "dict[str, ResourceDemand]":
        """
        Gets the host scores.

        Parameters:
            reqps (int): the reqps.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.

        Returns:
            dict[str, ResourceDemand]: the host scores.
        """

        dataToCache: "dict[str, list[float]]" = {}
        def parseEG(vnf: VNF, _depth: int) -> None:
            """
            Parses an EG.

            Parameters:
                vnf (VNF): the VNF.
                _depth (int): the depth.
                egID (str): the EG ID.
            """

            nonlocal dataToCache

            dataToCache[vnf["vnf"]["id"]] = [reqps]

        for eg in egs:
            traverseVNF(eg["vnfs"], parseEG, shouldParseTerminal=False)

        self._calibrate.predictAndCache(dataToCache)

        hostResourceData: "dict[str, ResourceDemand]" = {}
        for host, sfcs  in embeddingData.items():
            otherCPU: float = 0
            otherMemory: float = 0

            for vnfs in sfcs.values():
                for vnf, depth in vnfs:
                    divisor: int = 2**(depth-1)
                    effectiveReqps: float = reqps / divisor
                    demands: ResourceDemand = self._calibrate.getVNFResourceDemandForReqps(vnf, effectiveReqps)

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

    def getLinkScores(self, reqps: int, topology: Topology, egs: "list[EmbeddingGraph]", linkData: "dict[str, dict[str, float]]") -> "dict[str, float]":
        """
        Gets the link scores.

        Parameters:
            reqps (int): the reqps.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            linkData (dict[str, dict[str, float]]): the link data.

        Returns:
            dict[str, float]: the link scores.
        """

        linkScores: "dict[str, float]" = {}
        for eg in egs:
            totalLinkScore: float = 0
            for egLink in eg["links"]:
                links: "list[str]" = [egLink["source"]["id"]]
                links.extend(egLink["links"])
                links.append(egLink["destination"]["id"])
                divisor: int = egLink["divisor"]
                reqps: float = reqps / divisor

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    totalRequests: int = 0

                    if f"{source}-{destination}" in linkData:
                        for _key, data in linkData[f"{source}-{destination}"].items():
                            totalRequests += data * reqps
                    elif f"{destination}-{source}" in linkData:
                        for data in linkData[f"{destination}-{source}"].values():
                            totalRequests += data * reqps

                    bandwidth: float = [link["bandwidth"] for link in topology["links"] if (link["source"] == source and link["destination"] == destination) or (link["source"] == destination and link["destination"] == source)][0]

                    linkScore: float = self._getLinkScore(reqps, totalRequests, bandwidth)

                    totalLinkScore += linkScore

            linkScores[eg["sfcID"]] = totalLinkScore

        return linkScores

    def getSFCScores(self, data: "list[dict[str, dict[str, Union[int, float]]]]", topology: Topology, egs: "list[EmbeddingGraph]", embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]", linkData: "dict[str, dict[str, float]]" ) -> "list[list[Union[str, float]]]":
        """
        Gets the SFC scores.

        Parameters:
            data (list[dict[str, dict[str, Union[int, float]]]]): the data.
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the Embedding Graphs.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
            linkData (dict[str, dict[str, float]]): the link data.

        Returns:
            list[list[Union[str, float]]]: the SFC scores.
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

        for step in data:
            for sfc, sfcData in step.items():
                for vnf in vnfsInEGs[sfc]:
                    if vnf in dataToCache:
                        dataToCache[vnf].append(sfcData["reqps"])
                    else:
                        dataToCache[vnf] = [sfcData["reqps"]]

        self._calibrate = Calibrate()
        self._calibrate.predictAndCache(dataToCache)
        rows: "list[list[Union[str, float]]]" = []
        for step in data:
            """ hostResourceData: "dict[str, ResourceDemand]" = {}
            for host, sfcs  in embeddingData.items():
                otherCPU: float = 0
                otherMemory: float = 0

                for sfc, vnfs in sfcs.items():
                    for vnf, depth in vnfs:
                        divisor: int = 2**(depth-1)
                        reqps: float = (step[sfc]["reqps"] if sfc in step else 0) / divisor
                        demands: ResourceDemand = self._calibrate.getVNFResourceDemandForReqps(vnf, reqps)

                        vnfCPU: float = demands["cpu"]
                        vnfMemory: float = demands["memory"]
                        otherCPU += vnfCPU
                        otherMemory += vnfMemory

                hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)
            TUI.appendToSolverLog("Resource consumption of hosts calculated.") """

            hostVNFs: "dict[str, int]" = {}
            for host, sfcs in embeddingData.items():
                hostVNFs[host] = sum([len(vnfs) for vnfs in sfcs.values()])

            for sfc, sfcData in step.items():
                totalCPUScore: float = 0
                totalMemoryScore: float = 0
                totalLinkScore: float = 0
                eg: EmbeddingGraph = [graph for graph in egs if graph["sfcID"] == sfc][0]
                row: "list[Union[str, float]]" = []


                def parseVNF(vnf: VNF, depth: int) -> None:
                    """
                    Parses a VNF.

                    Parameters:
                        vnf (VNF): the VNF.
                        depth (int): the depth.
                    """

                    nonlocal totalCPUScore
                    nonlocal totalMemoryScore

                    divisor: int = 2**(depth-1)
                    reqps: float = sfcData["reqps"] / divisor
                    demands: ResourceDemand = self._calibrate.getVNFResourceDemandForReqps(vnf["vnf"]["id"], reqps)

                    vnfCPU: float = demands["cpu"]
                    vnfMemory: float = demands["memory"]

                    host: Host = [host for host in topology["hosts"] if host["id"] == vnf["host"]["id"]][0]
                    hostCPU: float = host["cpu"]
                    hostMemory: float = host["memory"]

                    cpuScore: float = self._getScore(vnfCPU, hostVNFs[vnf["host"]["id"]], hostCPU)
                    memoryScore: float = self._getScore(vnfMemory, hostVNFs[vnf["host"]["id"]], hostMemory)
                    totalCPUScore += cpuScore
                    totalMemoryScore += memoryScore

                traverseVNF(eg["vnfs"], parseVNF, shouldParseTerminal=False)

                TUI.appendToSolverLog(f"CPU Score: {totalCPUScore}. Memory Score: {totalMemoryScore}.")
                for egLink in eg["links"]:
                    links: "list[str]" = [egLink["source"]["id"]]
                    links.extend(egLink["links"])
                    links.append(egLink["destination"]["id"])
                    divisor: int = egLink["divisor"]
                    reqps: float = sfcData["reqps"] / divisor

                    for linkIndex in range(len(links) - 1):
                        source: str = links[linkIndex]
                        destination: str = links[linkIndex + 1]

                        totalRequests: int = 0

                        if f"{source}-{destination}" in linkData:
                            for key, data in linkData[f"{source}-{destination}"].items():
                                totalRequests += data * (step[key]["reqps"] if key in step else 0)
                        elif f"{destination}-{source}" in linkData:
                            for data in linkData[f"{destination}-{source}"].values():
                                totalRequests += data * (step[key]["reqps"] if key in step else 0)

                        bandwidth: float = [link["bandwidth"] for link in topology["links"] if (link["source"] == source and link["destination"] == destination) or (link["source"] == destination and link["destination"] == source)][0]

                        linkScore: float = self._getLinkScore(reqps, totalRequests, bandwidth)

                        totalLinkScore += linkScore

                TUI.appendToSolverLog(f"Link Score: {totalLinkScore}.")
                row.append(sfc)
                row.append(sfcData["reqps"])
                row.append(totalCPUScore)
                row.append(totalMemoryScore)
                row.append(totalLinkScore)
                row.append(sfcData["latency"])
                rows.append(row)

        return rows

    def _getScore(self, demand: float, totalVNFs: int, resource: float) -> float:
        """
        Gets the resource score.

        Parameters:
            demand (float): the demand.
            totalVNFs (int): the total VNFs.
            resource (float): the resource.

        Returns:
            float: the score.
        """

        return demand / (resource / totalVNFs)

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

        return (demand / totalDemand) * resource
