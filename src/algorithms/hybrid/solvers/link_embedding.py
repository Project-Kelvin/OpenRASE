"""
This defines the functions used for VNF link embedding.
"""

from typing import Union, cast
import networkx as nx
import heapq
from dijkstar import Graph, find_path
from shared.models import topology
from shared.models.sfc_request import SFCRequest
from algorithms.hybrid.utils.solvers import activationFunction
from algorithms.models.embedding import LinkData
from algorithms.utils.graphs import parseNodes
from constants.topology import SERVER, SFCC
from shared.models.topology import Link, Topology
from shared.models.embedding_graph import EmbeddingGraph, Optional
import numpy as np
from utils.tui import TUI


class HotCode:
    """
    This defines the hot code.
    """

    def __init__(self):
        """
        Initializes the hot code.
        """

        self.nodes: "dict[str, list[int]]" = {}
        self.sfcrs: "dict[str, list[int]]" = {}

    def addNode(self, name: str, length: int) -> None:
        """
        Adds a node.

        Parameters:
            name (str): the name.
            length (int): the length.
        """

        self.nodes[name] = [0] * length
        self.nodes[name][len(self.nodes) - 1] = 1

    def addSFC(self, name: str, length: int) -> None:
        """
        Adds an SFC.

        Parameters:
            name (str): the name.
            length (int): the length.
        """

        self.sfcrs[name] = [0] * length
        self.sfcrs[name][len(self.sfcrs) - 1] = 1

    def getNodeCode(self, name: str) -> "list[int]":
        """
        Gets the node code.

        Parameters:
            name (str): the name.

        Returns:
            list[int]: the code.
        """

        return self.nodes[name]

    def getSFCCode(self, name: str) -> "list[int]":
        """
        Gets the SFC code.

        Parameters:
            name (str): the name.

        Returns:
            list[int]: the code.
        """

        return self.sfcrs[name]


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

    def __init__(
        self,
        topology: Topology,
        sfcrs: "list[SFCRequest]",
        egs: "list[EmbeddingGraph]",
        predefinedWeights: "list[float]",
        weights: "list[float]",
        noOfNeurons: int,
        activation: str = "sin",
    ) -> None:
        """
        Initializes the link embedding.

        Parameters:
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the SFC requests.
            egs (list[EmbeddingGraph]): the EGs.
            predefinedWeights (list[float]): the predefined weights.
            weights (list[float]): the weights.
            noOfNeurons (int): the number of neurons in the hidden layer.
            activation (str): the type of activation function to apply.

        Returns:
            None
        """

        self._noOfNeurons: int = noOfNeurons
        self._sfcrs: list[SFCRequest] = sfcrs
        self._egs: list[EmbeddingGraph] = egs
        self._topology: Topology = topology
        self._graph: nx.Graph = self._constructGraph()
        self._pdWeights: list[float] = predefinedWeights
        self._weights: list[float] = weights
        self._hotCode: HotCode = HotCode()
        # cached once since these are re-derived from the (immutable) topology
        self._allLinks: list[str] = EmbedLinks.getLinks(self._topology)
        self._hostSet: set[str] = {host["id"] for host in self._topology["hosts"]} | {SFCC, SERVER}
        self._linkMap: dict[tuple[str, str], Link] = {}
        for link in self._topology["links"]:
            self._linkMap[(link["source"], link["destination"])] = link
            self._linkMap[(link["destination"], link["source"])] = link
        self._convertToHotCodes()
        self._hCost: dict[str, dict[str, dict[str, float]]] = {}
        self._linkIndex: dict[str, int] = {}
        self._activation: str = activation
        self._data: np.ndarray = self._predictCost()
        self._linkData: Optional[LinkData] = None


    def _isHost(self, node: str) -> bool:
        """
        Checks if the node is a host.

        Parameters:
            node (str): the node.

        Returns:
            bool: True if the node is a host, False otherwise.
        """

        return node in self._hostSet

    def _constructNP(self) -> np.ndarray:
        """
        Constructs the NumPy array.

        Returns:
            tuple[np.ndarray]: the NumPy array containing the input data.
        """

        rows: "list[list[int]]" = []
        for sfcr in self._sfcrs:
            for link in self._allLinks:
                row: "list[int]" = []
                row.extend(self._hotCode.getSFCCode(sfcr["sfcrID"]))
                row.extend(self._hotCode.getNodeCode(link))
                rows.append(row)
                self._linkIndex[f"{sfcr['sfcrID']}_{link}"] = len(rows) - 1

        return np.array(rows, dtype=np.float64)

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
            graph.add_edge(link["source"], link["destination"])
            graph.add_edge(link["destination"], link["source"])

        return graph

    @staticmethod
    def getLinks(topology: Topology) -> list[str]:
        """
        Gets the links.

        Parameters:
            topology (Topology): the topology.

        Returns:
            list[str]: the links.
        """

        links: list[str] = []
        for link in topology["links"]:
            links.append(f"{link['source']}_{link['destination']}")

        hosts: "list[str]" = [host["id"] for host in topology["hosts"]]
        hosts.append(SFCC)
        hosts.append(SERVER)

        switches: "list[int]" = [switch["id"] for switch in topology["switches"]]

        for switch in switches:
            for host in hosts:
                if f"{switch}_{host}" not in links:
                    links.append(f"{switch}_{host}")

        return links

    def _convertToHotCodes(self) -> None:
        """
        Converts the EGs to hot codes.

        Returns:
            None
        """

        for link in self._allLinks:
            self._hotCode.addNode(link, len(self._allLinks))

        sfcLength: int = len(self._sfcrs)

        for sfcr in self._sfcrs:
            self._hotCode.addSFC(sfcr["sfcrID"], sfcLength)

    def _predictCost(self) -> np.ndarray:
        """
        Builds the model.

        Returns:
            np.ndarray: the heuristic costs.
        """

        data = self._constructNP()
        npWeights = np.array(self._pdWeights, dtype=np.float64).reshape(-1, self._noOfNeurons if self._noOfNeurons > 0 else 1)
        heuristicCosts: np.ndarray = np.matmul(data, npWeights)
        heuristicCosts = abs(activationFunction(heuristicCosts, activation=self._activation))

        if self._noOfNeurons > 0:
            npWeights = np.array(self._weights, dtype=np.float64).reshape(-1, 1)
            heuristicCosts = np.matmul(heuristicCosts, npWeights)
            heuristicCosts = abs(activationFunction(heuristicCosts, activation=self._activation))

        return heuristicCosts

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

        index = self._linkIndex.get(f"{sfc}_{src}_{dst}")
        if index is None:
            index = self._linkIndex[f"{sfc}_{dst}_{src}"]

        return self._data[index]

    def _findPath(self, sfcID: str, source: str, destination: str) -> list[str]:
        """
        Finds the path using A*.

        Parameters:
            sfcID (str): the SFC ID.
            source (str): the source.
            destination (str): the destination.

        Returns:
            list[str]: the path.
        """

        startNode: Node = Node(source)
        counter: int = 0
        openHeap: "list[tuple[float, int, Node]]" = [
            (startNode.totalCost + startNode.hCost, counter, startNode)
        ]
        # best known total cost per node name, replaces the closedSet scan with an O(1) lookup
        bestCost: dict[str, float] = {source: 0.0}
        index: int = 0
        while len(openHeap) > 0:
            _, _, currentNode = heapq.heappop(openHeap)
            if currentNode.name == destination:
                path = []
                while currentNode is not None:
                    path.append(currentNode.name)
                    currentNode = currentNode.parent

                path.reverse()

                return path

            if currentNode.totalCost > bestCost.get(currentNode.name, float("inf")):
                index += 1
                continue

            if index == 0 or not self._isHost(currentNode.name):
                for neighbour in self._graph.adj[currentNode.name]:
                    totalCost = currentNode.totalCost + self._getHeuristicCost(
                        sfcID, currentNode.name, neighbour
                    )

                    if totalCost < bestCost.get(neighbour, float("inf")):
                        bestCost[neighbour] = totalCost
                        node: Node = Node(neighbour)
                        node.hCost = 0 if self._isHost(neighbour) else self._getHeuristicCost(
                            sfcID, neighbour, destination
                        )
                        node.parent = currentNode
                        node.totalCost = totalCost
                        counter += 1
                        heapq.heappush(openHeap, (node.totalCost + node.hCost, counter, node))

            index += 1

        return []

    def getLinkData(self) -> LinkData:
        """
        Gets the link data.

        Returns:
            LinkData : the link data.
        """

        if self._linkData is None:
            raise Exception("Link data is not available. Please run embedLinks() first.")

        return self._linkData

    def embedLinks(self, nodes: "dict[str, list[str]]", dijkstra: bool = False) -> "list[EmbeddingGraph]":
        """
        Embeds the links.

        Parameters:
            nodes (dict[str, list[str]]): the nodes to be linked.
            dijkstra (bool): whether to use Dijkstra's algorithm.

        Returns:
            list[EmbeddingGraph]: the EGs.
        """

        egsToRemove: "list[EmbeddingGraph]" = []
        for eg in self._egs:
            graph: Union[Graph, None] = None
            if dijkstra:
                graph = Graph()
                for link in self._topology["links"]:
                    graph.add_edge(link["source"], link["destination"], link["delay"] if "delay" in link and link["delay"] is not None else 1)
                    graph.add_edge(link["destination"], link["source"], link["delay"] if "delay" in link and link["delay"] is not None else 1)

            paths: "dict[str, list[str]]" = {}
            if "links" not in eg:
                eg["links"] = []

            sfcNodes, sfcDivisors = parseNodes(nodes[eg["sfcID"]])
            for nodeList, divisor in zip(sfcNodes, sfcDivisors):
                for i in range(len(nodeList) - 1):
                    if nodeList[i] == nodeList[i + 1]:
                        continue

                    srcDst: str = f"{nodeList[i]}-{nodeList[i + 1]}"
                    dstSrc: str = f"{nodeList[i + 1]}-{nodeList[i]}"

                    if srcDst not in paths and dstSrc not in paths:
                        try:
                            if dijkstra:
                                path = find_path(graph, nodeList[i], nodeList[i + 1]).nodes
                            else:
                                path = self._findPath(
                                    eg["sfcID"], nodeList[i], nodeList[i + 1]
                                )

                                if len(path) == 0:
                                    TUI.appendToSolverLog(f"Error: No path found between {nodeList[i]} and {nodeList[i + 1]}", True)
                                    egsToRemove.append(eg)
                                    continue

                            paths[srcDst] = path
                        except Exception as e:
                            TUI.appendToSolverLog(f"Error: {e}", True)
                            egsToRemove.append(eg)
                            continue

                        eg["links"].append(
                            {
                                "source": {"id": path[0]},
                                "destination": {"id": path[-1]},
                                "links": path[1:-1],
                            }
                        )

                    path = paths[srcDst] if srcDst in paths else paths[dstSrc]

                    for p in range(len(path) - 1):
                        matchedLink: Optional[Link] = self._linkMap.get((path[p], path[p + 1]))
                        if matchedLink is None:
                            raise Exception(f"No link found between {path[p]} and {path[p + 1]}")

                        link: Link = matchedLink
                        linkDelay: float = cast(float,(
                            (link["delay"] / divisor)
                            if "delay" in link and link["delay"] is not None
                            else 0.0
                        ))

                        if self._linkData is None:
                            self._linkData = cast(LinkData, {
                                f"{path[p]}-{path[p + 1]}": {str(eg["sfcID"]): (1.0 / float(divisor), linkDelay)}
                            })
                        else:
                            if f"{path[p]}-{path[p + 1]}" in self._linkData:
                                if (
                                    eg["sfcID"]
                                    in self._linkData[f"{path[p]}-{path[p + 1]}"]
                                ):
                                    pathData: tuple[float, float] = self._linkData[
                                        f"{path[p]}-{path[p + 1]}"
                                    ][eg["sfcID"]]
                                    divisors: float = pathData[0] + 1 / divisor
                                    delay: float = pathData[1] + linkDelay
                                    self._linkData[f"{path[p]}-{path[p + 1]}"][
                                        eg["sfcID"]
                                    ] = (divisors, delay)
                                else:
                                    self._linkData[f"{path[p]}-{path[p + 1]}"][
                                        eg["sfcID"]
                                    ] = (1 / divisor, linkDelay)
                            elif f"{path[p + 1]}-{path[p]}" in self._linkData:
                                if (
                                    eg["sfcID"]
                                    in self._linkData[f"{path[p + 1]}-{path[p]}"]
                                ):
                                    pathData: tuple[float, float] = self._linkData[
                                        f"{path[p + 1]}-{path[p]}"
                                    ][eg["sfcID"]]
                                    divisors: float = pathData[0] + 1 / divisor
                                    delay: float = pathData[1] + linkDelay
                                    self._linkData[f"{path[p + 1]}-{path[p]}"][
                                        eg["sfcID"]
                                    ] = (divisors, delay)
                                else:
                                    self._linkData[f"{path[p + 1]}-{path[p]}"][
                                        eg["sfcID"]
                                    ] = (1 / divisor, linkDelay)
                            else:
                                self._linkData[f"{path[p]}-{path[p + 1]}"] = {
                                    eg["sfcID"]: (1 / divisor, linkDelay)
                                }

        self._egs = [eg for eg in self._egs if eg["sfcID"] not in [eg["sfcID"] for eg in egsToRemove]]

        return self._egs
