"""
This defines a simple Dijkstra's algorithm to produce an Embedding Graph from a Forwarding Graph.
"""

import copy
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
from dijkstar import Graph, find_path
from constants.topology import SERVER, SFCC
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF


class SimpleDijkstraAlgorithm():
    """
    Class that implements the Simple Dijkstra's Algorithm.
    """

    _vnfResourceDemands: "dict[str, ResourceDemand]" = {}
    _fgs: "list[EmbeddingGraph]" = None
    _topology: Topology = None
    _nodeResourceUsage: "dict[str, ResourceDemand]" = {}
    _nodes: "dict[str, list[str]]" = {}

    def __init__(self, fg: "list[EmbeddingGraph]", topology: Topology, vnfResourceDemands: "dict[str, ResourceDemand]") -> None:
        """
        Initialize the Simple Dijkstra's Algorithm.
        """

        self._fgs = fg
        self._topology = topology
        self._vnfResourceDemands = vnfResourceDemands

        for graph in self._fgs:
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
                #nodeMemory: float = nodeResourceUsage["memory"] if nodeResourceUsage is not None else node["memory"]
                resourceDemand: ResourceDemand = copy.deepcopy(self._vnfResourceDemands[vnf["vnf"]["id"]])

                if divisor > 1:
                    fullResourceDemand: ResourceDemand = self._vnfResourceDemands[vnf["vnf"]["id"]]
                    resourceDemand["cpu"] = fullResourceDemand["cpu"] / divisor
                    #resourceDemand["memory"] = fullResourceDemand["memory"] / divisor

                if resourceDemand["cpu"] <= nodeCPU: #and resourceDemand["memory"] <= nodeMemory:
                    localNodeResourceUsage[node["id"]] = {
                        "cpu": nodeCPU - resourceDemand["cpu"],
                        #"memory": nodeMemory - resourceDemand["memory"]
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

    def _linkNodes(self, fg: EmbeddingGraph) -> EmbeddingGraph:
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
                    linkedEG: EmbeddingGraph = self._linkNodes(eg)
                    egs.append(linkedEG)
                    self._nodeResourceUsage = copy.deepcopy(localNodeResourceUsage)
                except Exception:
                    failedEGs.append(fg)
            else:
                failedEGs.append(fg)

        return (egs, failedEGs, self._nodeResourceUsage)