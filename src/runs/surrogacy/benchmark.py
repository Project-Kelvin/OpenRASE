"""
Defines script to benchmark CPU, memory and link usage against latency.
"""

import json
import os
import random
from time import sleep
from timeit import default_timer
import click
import pandas as pd
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.surrogacy.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.vnf_embedding import convertDFtoEGs, convertFGsToDF, getConfidenceValues
from algorithms.surrogacy.scorer import Scorer
from constants.topology import SERVER, SFCC
from mano.telemetry import Telemetry
from models.telemetry import HostData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.data import hostDataToFrame, mergeHostAndTrafficData
from utils.topology import generateFatTreeTopology
from utils.tui import TUI

egs: "list[EmbeddingGraph]" = [
    {
        "sfcID": "sfc1",
        "sfcrID": "sfcr1",
        "vnfs": {
            "host": {"id": "h1"},
            "vnf": {"id": "tm"},
            "next": {"host": {"id": SERVER}, "next": TERMINAL},
        },
        "links": [
            {"source": {"id": SFCC}, "destination": {"id": "h1"}, "links": ["s1"]},
            {"source": {"id": "h1"}, "destination": {"id": SERVER}, "links": ["s1"]},
        ],
    },
    {
        "sfcID": "sfc2",
        "sfcrID": "sfcr1",
        "vnfs": {
            "host": {"id": "h1"},
            "vnf": {"id": "tm"},
            "next": {"host": {"id": SERVER}, "next": TERMINAL},
        },
        "links": [
            {"source": {"id": SFCC}, "destination": {"id": "h1"}, "links": ["s1"]},
            {"source": {"id": "h1"}, "destination": {"id": SERVER}, "links": ["s1"]},
        ],
    },
    {
        "sfcID": "sfc3",
        "sfcrID": "sfcr1",
        "vnfs": {
            "host": {"id": "h1"},
            "vnf": {"id": "tm"},
            "next": {"host": {"id": SERVER}, "next": TERMINAL},
        },
        "links": [
            {"source": {"id": SFCC}, "destination": {"id": "h1"}, "links": ["s1"]},
            {"source": {"id": "h1"}, "destination": {"id": SERVER}, "links": ["s1"]},
        ],
    },
    {
        "sfcID": "sfc4",
        "sfcrID": "sfcr1",
        "vnfs": {
            "host": {"id": "h1"},
            "vnf": {"id": "tm"},
            "next": {"host": {"id": SERVER}, "next": TERMINAL},
        },
        "links": [
            {"source": {"id": SFCC}, "destination": {"id": "h1"}, "links": ["s1"]},
            {"source": {"id": "h1"}, "destination": {"id": SERVER}, "links": ["s1"]},
        ],
    },
]

trafficDuration: int = 120

artifactsDir: str = os.path.join(
    getConfig()["repoAbsolutePath"], "artifacts", "experiments", "surrogacy"
)

if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)

cpuDataPath: str = os.path.join(artifactsDir, "cpu.csv")
memoryDataPath: str = os.path.join(artifactsDir, "memory.csv")
controlDataPath: str = os.path.join(artifactsDir, "control.csv")


class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def generateRequests(self) -> None:

        self._orchestrator.sendRequests(egs)


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def __init__(self, orchestrator, trafficGenerator):
        super().__init__(orchestrator, trafficGenerator)
        self._data: pd.DataFrame = None

    def generateEmbeddingGraphs(self) -> None:
        """
        Generate the embedding graphs.
        """

        def updateTUI(sfcs: int) -> None:
            """
            Update the TUI.

            Parameters:
                sfcs (int): The number of SFCs.
            """

            TUI.appendToSolverLog("Starting traffic generation.")
            duration: int = trafficDuration
            elapsed: int = 0
            hostDataList: "list[HostData]" = []
            while elapsed < duration:
                try:
                    tele: Telemetry = self._orchestrator.getTelemetry()
                    start: float = default_timer()
                    hostData: HostData = tele.getHostData()
                    end: float = default_timer()
                    hostDataList.append(hostData)
                    teleDuration: int = round(end - start, 0)
                    elapsed += teleDuration
                except Exception as e:
                    TUI.appendToSolverLog(f"Error: {e}", True)

            try:
                trafficData: pd.DataFrame = self._trafficGenerator.getData(
                    f"{trafficDuration}s"
                )
                hostData: pd.DataFrame = hostDataToFrame(hostDataList)
                hostData = mergeHostAndTrafficData(hostData, trafficData)
                hostData["SFCs"] = sfcs + 1
                hostData["reqps"] = hostData["reqps"] / (sfcs + 1)

                if self._data is None:
                    self._data = hostData
                else:
                    self._data = pd.concat([self._data, hostData])
            except Exception as e:
                TUI.appendToSolverLog(f"Error: {e}", True)

            TUI.appendToSolverLog("Solver has finished.")

        for i in range(len(egs)):
            self._orchestrator.sendEmbeddingGraphs(egs[: i + 1])
            updateTUI(i)
            self._orchestrator.deleteEmbeddingGraphs(egs[: i + 1])
            sleep(10)
        TUI.exit()


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run headless.")
def benchmarkMemory(headless: bool) -> None:
    """
    This function benchmarks memory usage against latency.

    Parameters:
        headless (bool): Run headless.

    Returns:
        None
    """

    memoryTrafficDesign: "list[TrafficDesign]" = [
        [{"target": 400, "duration": f"{trafficDuration}s"}]
    ]

    memoryTopology: Topology = {
        "hosts": [{"memory": 1024, "id": "h1"}],
        "switches": [{"id": "s1"}],
        "links": [
            {"source": SFCC, "destination": "s1"},
            {"source": "s1", "destination": "h1"},
            {"source": SERVER, "destination": "s1"},
        ],
    }

    class MemorySolver(SFCSolver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            super().generateEmbeddingGraphs()
            self._data["memory"] = (
                self._data["memory"]
                / (1024 * 1024)
                / memoryTopology["hosts"][0]["memory"]
            )
            self._data.to_csv(memoryDataPath, index=False)

    sfcEmulator: SFCEmulator = SFCEmulator(SFCR, MemorySolver, headless)
    sfcEmulator.startTest(memoryTopology, memoryTrafficDesign)
    sfcEmulator.end()


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run headless.")
def benchmarkCPU(headless: bool) -> None:
    """
    This function benchmarks CPU usage against latency.

    Parameters:
        headless (bool): Run headless.

    Returns:
        None
    """

    cpuTrafficDesign: "list[TrafficDesign]" = [
        [{"target": 400, "duration": f"{trafficDuration}s"}]
    ]

    cpuTopology: Topology = {
        "hosts": [{"cpu": 0.5, "id": "h1"}],
        "switches": [{"id": "s1"}],
        "links": [
            {"source": SFCC, "destination": "s1"},
            {"source": "s1", "destination": "h1"},
            {"source": SERVER, "destination": "s1"},
        ],
    }

    class CPUSolver(SFCSolver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            super().generateEmbeddingGraphs()
            self._data["cpu"] = self._data["cpu"] / cpuTopology["hosts"][0]["cpu"]
            self._data.to_csv(cpuDataPath, index=False)

    sfcEmulator: SFCEmulator = SFCEmulator(SFCR, CPUSolver, headless)
    sfcEmulator.startTest(cpuTopology, cpuTrafficDesign)
    sfcEmulator.end()


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run headless.")
def benchmarkControl(headless: bool) -> None:
    """
    This function performs control benchmark.

    Parameters:
        headless (bool): Run headless.

    Returns:
        None
    """

    controlTrafficDesign: "list[TrafficDesign]" = [
        [{"target": 400, "duration": f"{trafficDuration}s"}]
    ]

    topology: Topology = {
        "hosts": [{"id": "h1"}],
        "switches": [{"id": "s1"}],
        "links": [
            {"source": SFCC, "destination": "s1"},
            {"source": "s1", "destination": "h1"},
            {"source": SERVER, "destination": "s1"},
        ],
    }

    class CPUSolver(SFCSolver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            super().generateEmbeddingGraphs()
            self._data.to_csv(controlDataPath, index=False)

    sfcEmulator: SFCEmulator = SFCEmulator(SFCR, CPUSolver, headless)
    sfcEmulator.startTest(topology, controlTrafficDesign)
    sfcEmulator.end()


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run headless.")
def benchmarkLinks(headless: bool) -> None:
    """
    This function benchmarks link usage against latency.

    Parameters:
        headless (bool): Run headless.

    Returns:
        None
    """

    topology: Topology = generateFatTreeTopology(4, 3, None, None)
    linksTrafficDesign: "list[TrafficDesign]" = [
        [
            {"target": 400, "duration": f"{trafficDuration}s"},
            {"target": 400, "duration": "15s"},
        ]
    ]

    fgs: "list[EmbeddingGraph]" = []

    with open(
        os.path.join("src", "runs", "surrogacy", "configs", "forwarding-graphs.json"),
        encoding="utf8",
    ) as f:
        fgs = [json.load(f)[0]]

    class LinkSFCR(SFCRequestGenerator):
        """
        SFC Request Generator.
        """

        def generateRequests(self) -> None:
            egsToSend: "list[EmbeddingGraph]" = []
            egCopy: EmbeddingGraph = fgs[0].copy()
            egCopy["sfcID"] = f"sfc{1}"
            egCopy["sfcrID"] = f"sfc{1}"
            egsToSend.append(egCopy)

            self._orchestrator.sendRequests(egsToSend)

    class LinksSolver(Solver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            """
            This function generates the embedding graphs.
            """

            try:
                while self._requests.empty():
                    pass
                requests: "list[EmbeddingGraph]" = []
                while not self._requests.empty():
                    requests.append(self._requests.get())
                    sleep(0.1)

                noOfGenes: int = getWeightLength(requests[0], topology)
                individual: "list[float]" = []
                eGraphs: "list[EmbeddingGraph]" = []

                while len(eGraphs) != len(requests):
                    for _ in range(noOfGenes):
                        individual.append(random.uniform(-1, 1))
                    weights: (
                        "tuple[list[float], list[float], list[float], list[float]]"
                    ) = getWeights(individual, requests, topology)
                    eGraphs, nodes, _ = convertDFtoEGs(
                        getConfidenceValues(
                            convertFGsToDF(requests, topology), weights[0], weights[1]
                        ),
                        requests,
                        topology,
                    )

                graphsToDeploy: "list[list[EmbeddingGraph]]" = []
                embedLinks: EmbedLinks = EmbedLinks(
                    topology, requests, weights[2], weights[3]
                )
                finalEGs: "list[EmbeddingGraph]" = embedLinks.embedLinks(nodes)

                graphsToDeploy.append(finalEGs)

                for _ in range(3):
                    for i, _ in enumerate(weights[2]):
                        weights[2][i] = random.uniform(-1, 1)
                    for i, _ in enumerate(weights[3]):
                        weights[3][i] = random.uniform(-1, 1)

                    embedLinks: EmbedLinks = EmbedLinks(
                        topology, requests, weights[2], weights[3]
                    )
                    finalEGs: "list[EmbeddingGraph]" = embedLinks.embedLinks(nodes)

                    graphsToDeploy.append(finalEGs)

                finalData: pd.DataFrame = pd.DataFrame()
                for index, graphs in enumerate(graphsToDeploy):

                    self._orchestrator.sendEmbeddingGraphs(graphs)
                    TUI.appendToSolverLog("Starting traffic generation.")
                    sleep(trafficDuration)

                    try:
                        trafficData: pd.DataFrame = self._trafficGenerator.getData(
                            f"{trafficDuration}s"
                        )

                        trafficData["_time"] = trafficData["_time"] // 1000000000

                        groupedTrafficData: pd.DataFrame = trafficData.groupby(
                            ["_time", "sfcID"]
                        ).agg(
                            reqps=("_value", "count"),
                            medianLatency=("_value", "median"),
                        )

                        for i, group in groupedTrafficData.groupby(level=0):
                            data: "dict[str, dict[str, float]]" = {
                                graph["sfcID"]: {
                                    "reqps": group.loc[i, graph["sfcID"]]["reqps"],
                                    "latency": group.loc[i, graph["sfcID"]][
                                        "medianLatency"
                                    ],
                                }
                                for graph in graphs
                            }

                            scorer: Scorer = Scorer()
                            scores: "dict[str, float]" = scorer.getLinkScores(
                                data, topology, graphs, embedLinks.getLinkData()
                            )

                            for key, value in scores.items():
                                groupedTrafficData.loc[(i, key), "linkScore"] = value

                        groupedTrafficData["run"] = index + 1
                        finalData = pd.concat([finalData, groupedTrafficData])
                    except Exception as e:
                        TUI.appendToSolverLog(f"Error: {e}", True)

                    TUI.appendToSolverLog("Solver has finished.")
                    self._orchestrator.deleteEmbeddingGraphs(graphs)
                    sleep(120)
                finalData.to_csv(os.path.join(artifactsDir, "links.csv"), index=False)
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)
            TUI.exit()

    sfcEmulator: SFCEmulator = SFCEmulator(LinkSFCR, LinksSolver, headless)
    sfcEmulator.startTest(topology, linksTrafficDesign)
    sfcEmulator.end()
