"""
This defines the functions used for VNF link embedding.
"""

from typing import Tuple
import networkx as nx
import heapq

from shared.models.sfc_request import SFCRequest
from algorithms.hybrid.constants.surrogate import BRANCH
from algorithms.hybrid.utils.solvers import activationFunction
from algorithms.utils.graphs import parseNodes
from constants.topology import SERVER, SFCC
from shared.models.topology import Link, Topology
from shared.models.embedding_graph import EmbeddingGraph
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
        noOfNeurons: int
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
        self._convertToHotCodes()
        self._hCost: dict[str, dict[str, dict[str, float]]] = {}
        self._links: list[str] = []
        self._data: np.ndarray = self._predictCost()
        self._linkData: dict[str, dict[str, float]] = {}

    def _isHost(self, node: str) -> bool:
        """
        Checks if the node is a host.

        Parameters:
            node (str): the node.

        Returns:
            bool: True if the node is a host, False otherwise.
        """

        return (
            node in [host["id"] for host in self._topology["hosts"]]
            or node == SFCC
            or node == SERVER
        )

    def _constructNP(self) -> np.ndarray:
        """
        Constructs the NumPy array.

        Returns:
            tuple[np.ndarray]: the NumPy array containing the input data.
        """

        rows: "list[list[int]]" = []
        links = EmbedLinks.getLinks(self._topology)
        for sfcr in self._sfcrs:
            for link in links:
                row: "list[int]" = []
                row.extend(self._hotCode.getSFCCode(sfcr["sfcrID"]))
                row.extend(self._hotCode.getNodeCode(link))
                rows.append(row)
                self._links.append(f"{sfcr['sfcrID']}_{link}")

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

        links = EmbedLinks.getLinks(self._topology)
        for link in links:
            self._hotCode.addNode(link, len(links))

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
        heuristicCosts = abs(activationFunction(heuristicCosts))

        if self._noOfNeurons > 0:
            npWeights = np.array(self._weights, dtype=np.float64).reshape(-1, 1)
            heuristicCosts = np.matmul(heuristicCosts, npWeights)
            heuristicCosts = abs(activationFunction(heuristicCosts))

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

        if f"{sfc}_{src}_{dst}" in self._links:
            index = self._links.index(f"{sfc}_{src}_{dst}")
        else:
            index = self._links.index(f"{sfc}_{dst}_{src}")

        return self._data[index]

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
        while len(openSet) > 0:
            currentNode: Node = heapq.heappop(openSet)
            if currentNode.name == destination:
                path = []
                while currentNode is not None:
                    path.append(currentNode.name)
                    currentNode = currentNode.parent

                path.reverse()

                return path

            if index == 0 or not self._isHost(currentNode.name):
                for neighbor in self._graph.adj[currentNode.name]:
                    node: Node = Node(neighbor)

                    if self._isHost(neighbor):
                        node.hCost = 0
                    else:
                        node.hCost = self._getHeuristicCost(
                            sfcID, neighbor, destination
                        )
                    node.parent = currentNode
                    node.totalCost = currentNode.totalCost + self._getHeuristicCost(
                        sfcID, currentNode.name, neighbor
                    )

                    if (
                        len(
                            [
                                closedSetNode
                                for closedSetNode in closedSet
                                if closedSetNode.name == neighbor
                                and node.totalCost >= closedSetNode.totalCost
                            ]
                        )
                        == 0
                    ):
                        heapq.heappush(openSet, node)

            index += 1
            closedSet.append(currentNode)

    def getLinkData(self) -> "dict[str, dict[str, float]]":
        """
        Gets the link data.

        Returns:
            dict[str, dict[str, float]] : the link data.
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
                            path = self._findPath(
                                eg["sfcID"], nodeList[i], nodeList[i + 1]
                            )
                            paths[srcDst] = path
                        except Exception as e:
                            TUI.appendToSolverLog(f"Error: {e}", True)
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
                        link: Link = [
                            topoLink
                            for topoLink in self._topology["links"]
                            if (
                                topoLink["source"] == path[p]
                                and topoLink["destination"] == path[p + 1]
                            )
                            or (
                                topoLink["source"] == path[p + 1]
                                and topoLink["destination"] == path[p]
                            )
                        ][0]
                        linkDelay: float = (
                            (link["delay"] / divisor)
                            if "delay" in link and link["delay"] is not None
                            else 0
                        )
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

        return self._egs
