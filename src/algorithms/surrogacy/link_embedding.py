"""
This defines the functions used for VNF link embedding.
"""

from timeit import default_timer
from typing import Tuple
import networkx as nx
import heapq
from algorithms.surrogacy.local_constants import BRANCH
from constants.topology import SERVER, SFCC
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
import tensorflow as tf
import numpy as np
from utils.tui import TUI
import pandas as pd

class HotCode:
    """
    This defines the hot code.
    """

    def __init__(self):
        """
        Initializes the hot code.
        """

        self.nodes = {}
        self.egs = {}

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

        self.egs[name] = len(self.egs)

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

        return self.egs[name]


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

class EmbedLinks:
    """
    This defines the logic used to embed links.
    """

    def __init__(self, topology: Topology, egs: "list[EmbeddingGraph]", weights: "list[float]", bias: "list[float]")  -> None:
        """
        Initializes the link embedding.

        Parameters:
            topology (Topology): the topology.
            egs (list[EmbeddingGraph]): the EGs.
            weights (list[float]): the weights.
            bias (list[float]): the bias.

        Returns:
            None
        """

        self._egs: "list[EmbeddingGraph]" = egs
        self._topology: Topology = topology
        self._graph: nx.Graph = self._constructGraph()
        self._weights: "list[float]" = weights
        self._bias: "list[float]" = bias
        self._hotCode: HotCode = HotCode()
        self._convertToHotCodes()
        self._hCost: "dict[str, dict[str, dict[str, float]]]" = {}
        self._data: pd.DataFrame = self._predictCost()
        self._linkData: "dict[str, dict[str, float]]" = {}

    def _isHost(self, node: str) -> bool:
        """
        Checks if the node is a host.

        Parameters:
            node (str): the node.

        Returns:
            bool: True if the node is a host, False otherwise.
        """

        return node in [host["id"] for host in self._topology["hosts"]] or node == SFCC or node == SERVER

    def _constructDF(self) -> pd.DataFrame:
        """
        Constructs the DataFrame.
        """

        links: "list[list[int]]" = []
        for link in self._topology["links"]:
            row: "list[int]" = []
            row.append(self._hotCode.getNodeCode(link["source"]))
            row.append(self._hotCode.getNodeCode(link["destination"]))

            links.append(row)

        hosts: "list[int]" = [self._hotCode.getNodeCode(host["id"]) for host in self._topology["hosts"]]
        hosts.append(self._hotCode.getNodeCode(SFCC))
        hosts.append(self._hotCode.getNodeCode(SERVER))

        switches: "list[int]" = [self._hotCode.getNodeCode(switch["id"]) for switch in self._topology["switches"]]

        for switch in switches:
            for host in hosts:
                links.append([switch, host])

        rows: "list[list[int]]" = []
        for eg in self._egs:
            sfcLinks: "list[list[int]]" = []
            for link in links:
                row: "list[int]" = []
                row.append(self._hotCode.getSFCCode(eg["sfcID"]))
                row.extend(link)
                sfcLinks.append(row)
            rows.extend(sfcLinks)

        return pd.DataFrame(rows, columns=["SFC", "Source", "Destination"])


    def _constructGraph(self) -> nx.Graph:
        """
        Constructs the graph.

        Parameters:
            topology (Topology): the topology.

        Returns:
            nx.Graph: the graph.
        """

        graph: nx.Graph = nx.Graph()

        for link in self._topology["links"]:
            graph.add_edge(
                link["source"], link["destination"])
            graph.add_edge(
                link["destination"], link["source"])

        return graph

    def _convertToHotCodes(self) -> None:
        """
        Converts the EGs to hot codes.

        Returns:
            None
        """

        self._hotCode.addNode(SFCC)
        self._hotCode.addNode(SERVER)

        for hosts in self._topology["hosts"]:
            self._hotCode.addNode(hosts["id"])

        for switch in self._topology["switches"]:
            self._hotCode.addNode(switch["id"])

        for eg in self._egs:
            self._hotCode.addSFC(eg["sfcID"])

    def _predictCost(self) -> pd.DataFrame:
        """
        Builds the model.

        Returns:
            The dataframe: pd.Dataframe.
        """

        data: pd.DataFrame = self._constructDF()
        normalizer = tf.keras.layers.Normalization(axis=1)
        normalizer.adapt(np.array(data))
        layers: "list[int]" = [3, 2, 1]

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(layers[0], )),
            normalizer,
            tf.keras.layers.Dense(layers[1], activation="relu"),
            tf.keras.layers.Dense(layers[2], activation="relu")
        ])

        index: int = 0
        wStartIndex: int = 0
        wEndIndex: int = layers[index] * layers[index + 1]
        bStartIndex: int = 0
        bEndIndex: int = layers[index + 1]
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.set_weights([
                    np.array(self._weights[wStartIndex:wEndIndex]).reshape(layers[index], layers[index + 1]),
                    np.array(self._bias[bStartIndex:bEndIndex]).reshape(layers[index + 1])
                ])
                index += 1
                if index < len(layers) - 1:
                    wStartIndex = wEndIndex
                    wEndIndex = wStartIndex + (layers[index] * layers[index + 1])
                    bStartIndex = bEndIndex
                    bEndIndex = bStartIndex + layers[index + 1]

        prediction = model.predict(np.array(data))

        return data.assign(Cost=prediction)

    def _getHeuristicCost(self, sfc: str, src: str, dst: str) -> float:
        """
        Gets the heuristic cost.

        Parameters:
            sfc (str): the SFC.
            src (str): the source.
            dst (str): the destination.

        Returns:
            float: the heuristic cost.
        """

        sfcCode: int = self._hotCode.getSFCCode(sfc)
        srcCode: int = self._hotCode.getNodeCode(src)
        dstCode: int = self._hotCode.getNodeCode(dst)

        row: "list[int]" = self._data.loc[
            (self._data["SFC"] == sfcCode) &
            (self._data["Source"] == srcCode) &
            (self._data["Destination"] == dstCode)]

        if row.empty:
            row = self._data.loc[
                (self._data["SFC"] == sfcCode) &
                (self._data["Source"] == dstCode) &
                (self._data["Destination"] == srcCode)]

        return row["Cost"].values[0]

    def _findPath(self, sfcID: str, source: str, destination: str) -> "list[str]":
        """
        Finds the path using A*.

        Parameters:
            sfcID (str): the SFC ID.
            source (str): the source.
            destination (str): the destination.

        Returns:
            list[str]: the path.
        """

        openSet: "list[Node]" = [Node(source)]
        closedSet: "list[Node]" = []
        index: int = 0
        TUI.appendToSolverLog(f"Finding path from {source} to {destination} for SFC {sfcID}.")
        while len(openSet) > 0:
            currentNode: Node = heapq.heappop(openSet)
            if currentNode.name == destination:
                path = []
                while currentNode is not None:
                    path.append(currentNode.name)
                    currentNode = currentNode.parent

                path.reverse()

                TUI.appendToSolverLog(f"Path found: {str(path)}")

                return path

            if index == 0 or not self._isHost(currentNode.name):
                for neighbor in self._graph.adj[currentNode.name]:
                    node: Node = Node(neighbor)

                    if self._isHost(neighbor):
                        node.hCost = 0
                    else:
                        node.hCost = self._getHeuristicCost(
                            sfcID,
                            neighbor,
                            destination)
                    node.parent = currentNode
                    node.totalCost = currentNode.totalCost + self._getHeuristicCost(
                        sfcID,
                        currentNode.name,
                        neighbor
                    )

                    if len([closedSetNode for closedSetNode in closedSet if closedSetNode.name == neighbor and node.totalCost >= closedSetNode.totalCost]) == 0:
                        heapq.heappush(openSet, node)

            index += 1
            closedSet.append(currentNode)

    def parseNodes(self, nodes: "list[str]") -> "Tuple[list[list[str]], list[int]]":
        """
        Parses the nodes.

        Parameters:
            nodes (list[str]): the nodes.

        Returns:
            Tuple[list[list[str]], list[int]]: the parsed nodes, the parsed divisors.
        """

        parsedNodes: "list[list[str]]" = []
        roots: "list[list[str]]" = []
        branch: "list[str]" = []
        connectingNode: str = None
        currentDivisor: int = 1
        divisors: "list[int]" = []
        parsedDivisors: "list[int]" = []

        for node in nodes:
            if node == BRANCH:
                roots.append(branch[:])
                parsedNodes.append(branch[:])
                parsedDivisors.append(currentDivisor)
                currentDivisor *= 2
                divisors.append(currentDivisor)
                connectingNode = branch[-1]
                branch = []
            elif node == SERVER:
                branch.append(node)
                parsedNodes.append(branch[:])
                parsedDivisors.append(currentDivisor)
                branch = []
                if len(roots) > 0:
                    lastRoot: "list[str]" = roots.pop()
                    currentDivisor = divisors.pop()
                    connectingNode = lastRoot[-1]
            else:
                if connectingNode:
                    parsedNodes.append([connectingNode, node])
                    parsedDivisors.append(currentDivisor)
                    connectingNode = None
                branch.append(node)

        return parsedNodes, parsedDivisors

    def getLinkData(self) -> "dict[str, dict[str, float]]":
        """
        Gets the link data.

        Returns:
            dict[str, float]: the link data.
        """

        return self._linkData

    def embedLinks(self, nodes: "dict[str, list[str]]") -> "list[EmbeddingGraph]":
        """
        Embeds the links.

        Parameters:
            nodes (dict[str, list[str]]): the nodes to be linked.

        Returns:
            list[EmbeddingGraph]: the EGs.
        """

        for eg in self._egs:
            if "links" not in eg:
                eg["links"] = []

            paths: "list[str]" = []
            sfcNodes, sfcDivisors = self.parseNodes(nodes[eg["sfcID"]])
            start: float = default_timer()

            for nodeList, divisor in zip(sfcNodes, sfcDivisors):
                for i in range(len(nodeList) - 1):
                    try:
                        if nodeList[i] == nodeList[i + 1]:
                            continue

                        path = self._findPath(eg["sfcID"], nodeList[i], nodeList[i + 1])

                        for p in range(len(path) - 1):
                            if f"{path[p]}-{path[p + 1]}" in self._linkData:
                                if eg["sfcID"] in self._linkData[f"{path[p]}-{path[p + 1]}"]:
                                    self._linkData[f"{path[p]}-{path[p + 1]}"][eg["sfcID"]] += 1/divisor
                                else:
                                    self._linkData[f"{path[p]}-{path[p + 1]}"][eg["sfcID"]] = 1/divisor
                            elif f"{path[p + 1]}-{path[p]}" in self._linkData:
                                if eg["sfcID"] in self._linkData[f"{path[p + 1]}-{path[p]}"]:
                                    self._linkData[f"{path[p + 1]}-{path[p]}"][eg["sfcID"]] += 1/divisor
                                else:
                                    self._linkData[f"{path[p + 1]}-{path[p]}"][eg["sfcID"]] = 1/divisor
                                self._linkData[f"{path[p + 1]}-{path[p]}"][eg["sfcID"]] += 1/divisor
                            else:
                                self._linkData[f"{path[p]}-{path[p + 1]}"] = {
                                    eg["sfcID"]: 1/divisor
                                }


                        if f"{nodeList[i]}-{nodeList[i + 1]}" in paths:
                            continue

                        paths.append(f"{nodeList[i]}-{nodeList[i + 1]}")

                    except Exception as e:
                        TUI.appendToSolverLog(f"Error: {e}", True)
                        continue

                    eg["links"].append({
                        "source": {"id": path[0]},
                        "destination": {"id": path[-1]},
                        "links": path[1:-1],
                        "divisor": divisor
                    })
            end: float = default_timer()

            TUI.appendToSolverLog(f"Link Embedding Time for EG {eg['sfcID']}: {end - start}s")

        return self._egs
