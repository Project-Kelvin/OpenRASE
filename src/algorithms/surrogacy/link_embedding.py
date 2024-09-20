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
        self._hCost = 0
        self._totalCost = 0
        self._parent = None

    @property
    def hCost(self):
        """
        The heuristic cost of the node.
        """

        return self._hCost

    @hCost.setter
    def hCost(self, value):
        """
        Sets the heuristic cost of the node.
        """

        self._hCost = value

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

    @property
    def totalCost(self):
        """
        The total cost of the node.
        """

        return self._totalCost

    @totalCost.setter
    def totalCost(self, value):
        """
        Sets the total cost of the node.
        """

        self._totalCost = value

    def __lt__(self, other):
        return self._totalCost + self.hCost < other.totalCost + other.hCost

    def __eq__(self, name):
        return self.name == name

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

    hCost = {
        "src": 10,
        "s1": 5,
        "s2": 4,
        "s3": 3,
        "s4": 2,
        "s5": 3,
        "s6": 6,
        "s7": 8,
        "s8": 9,
        "h1": 4,
        "dst": 0
    }

    cost ={
        "src-s1": 2,
        "src-s2": 1,
        "s1-s6": 3,
        "s1-s3": 1,
        "s2-s3": 2,
        "s2-s5": 4,
        "s3-s7": 4,
        "s3-s4": 3,
        "s6-s7": 1,
        "s7-s8": 2,
        "s4-dst": 1,
        "s4-s5": 2,
        "s5-h1": 1
    }

    while len(openSet) > 0:
        currentNode: Node = heapq.heappop(openSet)
        print("Current Node: ", currentNode.name, currentNode.totalCost)
        if currentNode.name == destination:
            path = []
            while currentNode is not None:
                path.append(currentNode.name)
                currentNode = currentNode.parent

            path.reverse()

            return path

        for neighbor in graph.adj[currentNode.name]:
            if "h" in neighbor:
                continue

            node: Node = Node(neighbor)
            node.hCost = hCost[neighbor]
            node.parent = currentNode
            node.totalCost = currentNode.totalCost + (cost[f"{currentNode.name}-{neighbor}"] if f"{currentNode.name}-{neighbor}" in cost else cost[f"{neighbor}-{currentNode.name}"])
            print("Neighbor: ", neighbor, node.totalCost)
            print(currentNode.totalCost, cost[f"{currentNode.name}-{neighbor}"] if f"{currentNode.name}-{neighbor}" in cost else cost[f"{neighbor}-{currentNode.name}"])
            if len([closedSetNode for closedSetNode in closedSet if closedSetNode.name == neighbor and node.totalCost >= closedSetNode.totalCost]) == 0:
                heapq.heappush(openSet, node)
                print("Open List: ", [(node.name, node.hCost + node.totalCost, node.hCost, node.totalCost) for node in openSet])

        closedSet.append(currentNode)
        print("Closed List: ", [(node.name, node.hCost + node.totalCost) for node in closedSet])

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
