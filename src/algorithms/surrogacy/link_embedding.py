"""
This defines the functions used for VNF link embedding.
"""

from typing import Union
import networkx as nx
import heapq
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from runs.test import SFCR
from utils.topology import generateFatTreeTopology
import tensorflow as tf
import numpy as np

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

class HotCode:
    """
    This defines the hot code.
    """

    def __init__(self):
        """
        Initializes the hot code.
        """

        self.nodes = {}
        self.sfcs = {}

    def addNode(self, name: str):
        """
        Adds a node.

        Parameters:
            name (str): the name.
        """

        self.nodes[name] = len(self.nodes)

    def addSFC(self, name: str):
        """
        Adds an SFC.

        Parameters:
            name (str): the name.
        """

        self.sfcs[name] = len(self.sfcs)

    def getNodeCode(self, name: str) -> int:
        """
        Gets the node code.

        Parameters:
            name (str): the name.

        Returns:
            int: the code.
        """

        return self.nodes[name]

    def getSFCCode(self, name: str) -> int:
        """
        Gets the SFC code.

        Parameters:
            name (str): the name.

        Returns:
            int: the code.
        """

        return self.sfcs[name]


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

def findPath(graph: nx.Graph, source: str, destination: str, sfcID: str, hotCode: HotCode, weights: "list[float]") -> "list[str]":
    """
    Finds the path using A*.

    Parameters:
        graph (nx.Graph): the graph.
        source (str): the source.
        destination (str): the destination.
        sfcID (str): the SFC ID.
        hotCode (HotCode): the hot code.

    Returns:
        list[str]: the path.
    """

    openSet: "list[Node]" = [Node(source)]
    closedSet: "list[Node]" = []

    while len(openSet) > 0:
        currentNode: Node = heapq.heappop(openSet)
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
            node.hCost = getHeuristicCost(hotCode.getSFCCode(sfcID), hotCode.getNodeCode(neighbor), hotCode.getNodeCode(destination))
            node.parent = currentNode
            node.totalCost = currentNode.totalCost + getHeuristicCost(
                hotCode.getSFCCode(sfcID),
                hotCode.getNodeCode(currentNode.name),
                hotCode.getNodeCode(neighbor)
            )

            if len([closedSetNode for closedSetNode in closedSet if closedSetNode.name == neighbor and node.totalCost >= closedSetNode.totalCost]) == 0:
                heapq.heappush(openSet, node)

        closedSet.append(currentNode)

def getHeuristicCost(sfc: str, src: str, dst: str, weights: "list[float]", bias: "list[float]") -> float:
    """
    Gets the heuristic cost.

    Parameters:
        sfc (str): the SFC.
        src (str): the source.
        dst (str): the destination.
        weights (list[float]): the weights.
        bias (list[float]): the bias.

    Returns:
        float: the heuristic cost.
    """

    layers: "list[int]" = [3, 1]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(layers[0], )),
        tf.keras.layers.Dense(layers[1], activation='relu')
    ])

    index: int = 0
    startIndex: int = 0
    endIndex: int = layers[index]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.set_weights([
                np.array(weights[startIndex:endIndex]).reshape(layers[index], layers[index + 1]),
                np.array([bias[index]])
            ])
            index += 1
            if index < len(layers) - 1:
                startIndex = endIndex
                endIndex = startIndex + layers[index]

    prediction = model.predict(np.array([sfc, src, dst]).reshape(1, 3))

    return prediction


def convertToHotCodes(hotCode: HotCode, sfcrs: "list[Union[SFCR, EmbeddingGraph]]", topology: Topology) -> None:
    """
    Converts the SFCRs to hot codes.

    Parameters:
        hotCode (HotCode): the hot code.
        sfcrs (list[Union[SFCR, EmbeddingGraph]]): the SFCRs.
        topology (Topology): the topology.

    Returns:
        None
    """

    map(hotCode.addNode, [node["id"] for node in (topology["hosts"] + topology["switches"])])
    map(hotCode.addSFC, [sfcr["sfcID"] for sfcr in sfcrs])

sg: nx.Graph = nx.Graph()
sg.add_edge("a", "b")
sg.add_edge("a", "c")
sg.add_edge("c", "d")
sg.add_edge("d", "e")
sg.add_edge("e", "z")
sg.add_edge("b", "f")
sg.add_edge("f", "z")
sg.add_edge("b", "e")
sg.add_edge("c", "e")


#print(findPath(sg, "a", "z"))
hCode = HotCode()
hCode.addNode("a")
hCode.addNode("z")
hCode.addSFC("sfc1")
print(getHeuristicCost(hCode.getSFCCode("sfc1"), hCode.getNodeCode("a"),hCode.getNodeCode("z"), [1, 1, 1], [1]))
