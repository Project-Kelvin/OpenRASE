"""
This defines the functions used for VNF link embedding.
"""

import networkx as nx
import heapq
from shared.models.topology import Topology
from utils.topology import generateFatTreeTopology

def constructGraph(topology: Topology) -> nx.Graph:
    """
    Constructs the graph.

    Parameters:
        topology (Topology): the topology.

    Returns:
        nx.Graph: the graph.
    """

    graph: nx.Graph = nx.Graph()

    for link in topology["links"]:
        graph.add_edge(
            link["source"], link["destination"])
        graph.add_edge(
            link["destination"], link["source"])

    return graph

topo = generateFatTreeTopology(4, 100, 100, 100)

class Node:
    """
    This defines a node.
    """

    def __init__(self, name: str):
        self.name = name
        self._cost = 0
        self._parent = None

    @property
    def cost(self):
        """
        The cost of the node.
        """

        return self._cost

    @cost.setter
    def cost(self, value):
        """
        Sets the cost of the node.
        """

        self._cost = value

    @property
    def parent(self):
        """
        The parent of the node.
        """

        return self._parent

    @parent.setter
    def parent(self, value):
        """
        Sets the parent of the node.
        """

        self._parent = value

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.name == other.name

def findPath(graph: nx.Graph, source: str, destination: str) -> "list[str]":
    """
    Finds the path using A*.

    Parameters:
        graph (nx.Graph): the graph.
        source (str): the source.
        destination (str): the destination.

    Returns:
        list[str]: the path.
    """

    openSet: "list[Node]" = [Node(source)]
    closedSet: "list[Node]" = []

    while len(openSet) > 0:
        currentNode: Node = heapq.heappop(openSet)

        for neighbor in graph.adj[currentNode.name]:
            if neighbor == destination:
                path = [destination]
                while currentNode is not None:
                    path.append(currentNode.name)
                    currentNode = currentNode.parent

                path.reverse()

                return path

            if neighbor in [openSetNode.name for openSetNode in openSet]:
                continue

            if neighbor == currentNode.name:
                continue

            if "h" in neighbor:
                continue

            if neighbor in [closedSetNode.name for closedSetNode in closedSet]:
                continue

            node: Node = Node(neighbor)
            node.cost = 1
            node.parent = currentNode
            heapq.heappush(openSet, node)

        closedSet.append(currentNode)

sg: nx.Graph = nx.Graph()
sg.add_edge("src", "s1")
sg.add_edge("src", "s2")
sg.add_edge("s1", "s6")
sg.add_edge("s1", "s3")
sg.add_edge("s2", "s3")
sg.add_edge("s2", "s5")
sg.add_edge("s3", "s7")
sg.add_edge("s3", "s4")
sg.add_edge("s6", "s7")
sg.add_edge("s7", "s8")
sg.add_edge("s4", "dst")
sg.add_edge("s4", "s5")
sg.add_edge("s5", "h1")

print(findPath(sg, "src", "dst"))
