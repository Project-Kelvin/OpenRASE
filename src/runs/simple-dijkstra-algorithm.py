"""
This defines a simple Dijkstra's algorithm to produce an EMbedding Graph from a Forwarding Graph.
"""

from shared.constants.embedding_graph import TERMINAL
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
    _fg: EmbeddingGraph = None
    _topology: Topology = None
    _nodeResourceUsage: "dict[str, ResourceDemand]" = {}
    _eg: EmbeddingGraph = None
    _nodes: "list[str]" = []

    def __init__(self, fg: EmbeddingGraph, topology: Topology, vnfResourceDemands: "dict[str, ResourceDemand]") -> None:
        self._fg = fg
        self._topology = topology
        self._eg = fg.copy()
        self._nodes.append(SFCC)
        self._vnfResourceDemands = vnfResourceDemands

    def findNode(self):
        """
        Find the node with the required resources.
        """

        def traverseNodes(vnf: VNF):
            """
            Traverse the nodes to find the node with the required resources.
            """

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                self._nodes.append(SERVER)

                return

            for node in self._topology["hosts"]:
                nodeResourceUsage: ResourceDemand = self._nodeResourceUsage[node["id"]
                                                                            ] if node["id"] in self._nodeResourceUsage else None
                nodeCPU: float = nodeResourceUsage["cpu"] if nodeResourceUsage is not None else node["cpu"]
                nodeMemory: float = nodeResourceUsage["memory"] if nodeResourceUsage is not None else node["memory"]
                resourceDemand: ResourceDemand = self._vnfResourceDemands[vnf["vnf"]["id"]]

                if resourceDemand["cpu"] <= nodeCPU and resourceDemand["memory"] <= nodeMemory:
                    self._nodeResourceUsage[node["id"]] = {
                        "cpu": nodeCPU - resourceDemand["cpu"],
                        "memory": nodeMemory - resourceDemand["memory"]
                    }

                    if self._nodes[-1] != node["id"]:
                        self._nodes.append(node["id"])

                    vnf["host"] = {
                        "id": node["id"],
                    }

                    break

        traverseVNF(self._fg["vnfs"], traverseNodes)

    def linkNodes(self):
        """
        Link nodes
        """

        graph = Graph()

        if "links" not in self._eg:
            self._eg["links"] = []

        for link in self._topology["links"]:
            graph.add_edge(
                link["source"], link["destination"], link["bandwidth"])
            graph.add_edge(
                link["destination"], link["source"], link["bandwidth"])

        for i in range(len(self._nodes) - 1):
            path = find_path(graph, self._nodes[i], self._nodes[i + 1])
            self._eg["links"].append({
                "source": {"id": path.nodes[0]},
                "destination": {"id": path.nodes[-1]},
                "links": path.nodes[1:-1]
            })

    def run(self) -> EmbeddingGraph:
        """
        Run the Simple Dijkstra's Algorithm.
        """

        self.findNode()
        self.linkNodes()

        return self._eg


topo: Topology = {
    "hosts": [
        {
            "id": "h1",
            "cpu": 1,
            "memory": 1024
        },
        {
            "id": "h2",
            "cpu": 1,
            "memory": 1024
        }
    ],
    "switches": [
        {
            "id": "s1"
        },
        {
            "id": "s2"
        }
    ],
    "links": [
        {
            "source": SFCC,
            "destination": "s1",
            "bandwidth": 1000
        },
        {
            "source": "s1",
            "destination": "h1",
            "bandwidth": 1000
        },
        {
            "source": "s1",
            "destination": "s2",
            "bandwidth": 1000
        },
        {
            "source": "s2",
            "destination": "h2",
            "bandwidth": 1000
        },
        {
            "source": "s2",
            "destination": SERVER,
            "bandwidth": 1000
        }
    ]
}

forwardingGraph: EmbeddingGraph = {
    "sfcID": "sfc1",
    "vnfs": {
        "vnf": {
            "id": "waf"
        },
        "next": {
            "vnf": {
                "id": "ids"
            },
            "next": {
                "vnf": {
                    "id": "nat"
                },
                "next": {
                    "vnf": {
                        "id": "lb",
                    },
                    "next": {
                        "host": {
                            "id": SERVER,
                        },
                        "next": TERMINAL
                    }
                }
            }
        }
    }
}
vnfDemand: "dict[str, ResourceDemand]" = {
    "waf": {
        "cpu": 0.5,
        "memory": 512
    },
    "ids": {
        "cpu": 0.5,
        "memory": 512
    },
    "nat": {
        "cpu": 0.5,
        "memory": 512
    },
    "lb": {
        "cpu": 0.5,
        "memory": 512
    }
}

sda = SimpleDijkstraAlgorithm(forwardingGraph, topo, vnfDemand)
print(sda.run())
