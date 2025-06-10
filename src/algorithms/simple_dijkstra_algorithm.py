"""
This defines a simple Dijkstra's algorithm to produce an Embedding Graph from a Forwarding Graph.
"""

import copy
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from dijkstar import Graph, find_path
from calibrate.demand_predictor import DemandPredictor
from constants.topology import SERVER, SFCC
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF


class SimpleDijkstraAlgorithm():
    """
    Class that implements the Simple Dijkstra's Algorithm.
    """

    def __init__(self, fg: "list[EmbeddingGraph]", topology: Topology, maxTarget: int) -> None:
        """
        Initialize the Simple Dijkstra's Algorithm.

        Parameters:
            fg (list[EmbeddingGraph]): the Forwarding Graphs.
            topology (Topology): the topology.
            maxTarget (int): the maximum target for the algorithm.
        """

        self._fgs: "list[EmbeddingGraph]" = fg
        self._topology: Topology = topology
        self._nodeResourceUsage: "dict[str, ResourceDemand]" = {}
        self._nodes: "dict[str, list[str]]" = {}
        self._demandPredictor: DemandPredictor = DemandPredictor()
        self._maxTarget: int = maxTarget

        for graph in self._fgs:
            graph["sfcID"] = graph["sfcrID"]
            self._nodes[graph["sfcID"]]= [SFCC]

    def _findNode(self, fg: EmbeddingGraph) -> "tuple[EmbeddingGraph, bool, dict[str, ResourceDemand]]":
        """
        Find the node with the required resources.

        Parameters:
            fg (EmbeddingGraph): the Forwarding Graph.

        Returns:
            tuple[EmbeddingGraph, bool, dict[str, ResourceDemand]]:
                the Embedding Graph, True if all VNFs are deployed, False otherwise, and node resource usage.
        """

        areAllVNFsDeployed: "list[bool]" = [True]
        eg: EmbeddingGraph = copy.deepcopy(fg)

        localNodeResourceUsage: "dict[str, ResourceDemand]" = copy.deepcopy(self._nodeResourceUsage)
        def traverseNodes(vnf: VNF, depth: int, areAllVNFsDeployed: "list[bool]"):
            """
            Traverse the nodes to find the node with the required resources.
            """

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                self._nodes[eg["sfcID"]].append(SERVER)

                return

            isNodeAvailable: bool = False

            divisor: int = 2**(depth-1)

            for node in self._topology["hosts"]:
                nodeResourceUsage: ResourceDemand = localNodeResourceUsage[node["id"]
                                                                            ] if node["id"] in localNodeResourceUsage else None
                nodeCPU: float = nodeResourceUsage["cpu"] if nodeResourceUsage is not None else node["cpu"]
                resourceDemand: ResourceDemand = (
                    self._demandPredictor.getResourceDemandsOfVNF(
                        vnf["vnf"]["id"], self._maxTarget / divisor
                    )
                )

                if resourceDemand["cpu"] <= nodeCPU:
                    localNodeResourceUsage[node["id"]] = {
                        "cpu": nodeCPU - resourceDemand["cpu"],
                    }

                    if self._nodes[eg["sfcID"]][-1] != node["id"]:
                        self._nodes[eg["sfcID"]].append(node["id"])

                    vnf["host"] = {
                        "id": node["id"],
                    }

                    isNodeAvailable = True

                    break

            if isNodeAvailable is False:
                areAllVNFsDeployed[0] = False

        traverseVNF(eg["vnfs"], traverseNodes, areAllVNFsDeployed)

        return (eg, areAllVNFsDeployed[0], localNodeResourceUsage)

    def linkNodes(self, fg: EmbeddingGraph) -> EmbeddingGraph:
        """
        Link nodes

        Parameters:
            fg (EmbeddingGraph): The Forwarding Graph.

        Returns:
            EmbeddingGraph: The Embedding Graph.
        """

        graph = Graph()
        nodePair: "list[str]" = []
        eg: EmbeddingGraph = copy.deepcopy(fg)

        if "links" not in eg:
            eg["links"] = []

        for link in self._topology["links"]:
            graph.add_edge(
                link["source"], link["destination"], link["bandwidth"])
            graph.add_edge(
                link["destination"], link["source"], link["bandwidth"])

        for i in range(len(self._nodes[eg["sfcID"]]) - 1):
            srcDst: str = f"{self._nodes[eg['sfcID']][i]}-{self._nodes[eg['sfcID']][i + 1]}"
            dstSrc: str = f"{self._nodes[eg['sfcID']][i + 1]}-{self._nodes[eg['sfcID']][i]}"
            if srcDst not in nodePair and dstSrc not in nodePair:
                nodePair.append(srcDst)
                nodePair.append(dstSrc)
                path = find_path(graph, self._nodes[eg["sfcID"]][i], self._nodes[eg["sfcID"]][i + 1])

                eg["links"].append({
                    "source": {"id": path.nodes[0]},
                    "destination": {"id": path.nodes[-1]},
                    "links": path.nodes[1:-1]
                })

        return eg

    def run(self) -> "tuple[EmbeddingGraph, EmbeddingGraph, ResourceDemand]":
        """
        Run the Simple Dijkstra's Algorithm.

        Returns:
            tuple[EmbeddingGraph, EmbeddingGraph, ResourceDemand]: the successful and failed Embedding Graphs,
            and the resources used in nodes.
        """

        egs: "list[EmbeddingGraph]" = []
        failedEGs: "list[EmbeddingGraph]" = []

        for fg in self._fgs:
            eg, isEmbeddable, localNodeResourceUsage = self._findNode(fg)
            if isEmbeddable:
                try:
                    linkedEG: EmbeddingGraph = self.linkNodes(eg)
                    egs.append(linkedEG)
                    self._nodeResourceUsage = copy.deepcopy(localNodeResourceUsage)
                except Exception:
                    failedEGs.append(fg)
            else:
                failedEGs.append(fg)

        return (egs, failedEGs, self._nodeResourceUsage)
