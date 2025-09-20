"""
This runs the Simple Dijsktra Algorithm.
"""

import json
import os
from time import sleep
from timeit import default_timer
from typing import Any
import click
import pandas as pd
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.greedy_dijkstra_algorithm import GreedyDijkstraAlgorithm
from algorithms.hybrid.genesis import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile
from utils.tui import TUI


config: Config = getConfig()
directory = os.path.join(
    config['repoAbsolutePath'], "artifacts", "experiments", "simple_dijkstra_algorithm"
)

if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(directory, "genesis_comparison")

if not os.path.exists(directory):
    os.makedirs(directory)

@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.

    Returns:
        None
    """

    experimentsIncludeFilter: list[dict[str, Any]] = []
    experimentsExcludeFilter: list[dict[str, Any]] = []
    experimentPriority: list[str] = []
    experimentsToRun: list[dict[str, Any]] = []

    for noOfCopy in [8, 12]:
        for trafficScale in [0.1, 0.2]:
            for trafficPattern in [False, True]:
                for linkBandwidth in [10, 5]:
                    for noOfCPUs in [2, 1, 0.5]:
                        experimentsToRun.append(
                            {
                                "name": f"{noOfCopy}_{trafficScale}_{trafficPattern}_{linkBandwidth}_{noOfCPUs}",
                                "noOfCopies": noOfCopy,
                                "trafficScale": trafficScale,
                                "trafficPattern": trafficPattern,
                                "linkBandwidth": linkBandwidth,
                                "noOfCPUs": noOfCPUs,
                                "memory": 5120,
                            }
                        )

    if len(experimentPriority) > 0:
        experimentsToRun = sorted(
            experimentsToRun,
            key=lambda x: (
                experimentPriority.index(x["name"])
                if x["name"] in experimentPriority
                else len(experimentPriority)
            ),
        )

    for exp in experimentsToRun:
        if (
            len(experimentsIncludeFilter) > 0
            and (
                exp["noOfCopies"],
                exp["trafficScale"],
                exp["trafficPattern"],
                exp["linkBandwidth"],
                exp["noOfCPUs"],
            )
            not in experimentsIncludeFilter
        ):
            continue

        if (
            len(experimentsExcludeFilter) > 0
            and (
                exp["noOfCopies"],
                exp["trafficScale"],
                exp["trafficPattern"],
                exp["linkBandwidth"],
                exp["noOfCPUs"],
            )
            in experimentsExcludeFilter
        ):
            continue

        class SFCRGen(SFCRequestGenerator):
            """
            Class to generate FG Requests.
            """

            def __init__(self, orchestrator: Orchestrator) -> None:
                """
                Initialize the SFCRGen class.
                """

                super().__init__(orchestrator)
                with open(
                    os.path.join(
                        getConfig()["repoAbsolutePath"],
                        "src",
                        "runs",
                        "greedy_dijkstra_algorithm",
                        "configs",
                        "forwarding-graphs.json",
                    ),
                    "r",
                    encoding="utf8",
                ) as f:
                    self.fgrs = json.load(f)

            def generateRequests(self) -> "list[EmbeddingGraph]":
                """
                Generate the FG Requests.
                """

                fgrsToSend: "list[SFCRequest]" = []

                for i, fgr in enumerate(self.fgrs):
                    for c in range(exp["noOfCopies"]):
                        fgrToSend: SFCRequest = fgr.copy()
                        fgrToSend["sfcrID"] = f"sfcr{i}-{c}"
                        fgrsToSend.append(fgrToSend)

                self._orchestrator.sendRequests(fgrsToSend)

        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesignFromFile(
                os.path.join(
                    f"{getConfig()['repoAbsolutePath']}",
                    "src",
                    "runs",
                    "greedy_dijkstra_algorithm",
                    "data",
                    "requests.csv",
                ),
                exp["trafficScale"],
                4,
                False,
                exp["trafficPattern"],
            )
        ]

        topology: Topology = generateFatTreeTopology(
            4, exp["linkBandwidth"], exp["noOfCPUs"], exp["memory"], 1
        )

        class GDASolver(Solver):
            """
            Class to run the GDA algorithm.
            """

            def generateEmbeddingGraphs(self):
                """
                Generate the embedding graphs.
                """

                try:
                    while self._requests.empty():
                        pass
                    requests: "list[EmbeddingGraph]" = []
                    while not self._requests.empty():
                        requests.append(self._requests.get())
                        sleep(0.1)

                    maxTarget: int = max(trafficDesign[0], key=lambda x: x["target"])[
                        "target"
                    ]
                    sda = GreedyDijkstraAlgorithm(requests, topology, maxTarget)
                    start: float = default_timer()
                    fgs, failedFGs, _nodes = sda.run()
                    end: float = default_timer()
                    executionTime: float = end - start
                    acceptanceRatio: float = len(fgs) / len(fgs + failedFGs)

                    self._orchestrator.sendEmbeddingGraphs(fgs)
                    trafficDuration: int = calculateTrafficDuration(trafficDesign[0])
                    TUI.appendToSolverLog(f"Waiting for {trafficDuration}s.")
                    sleep(trafficDuration)

                    trafficData: pd.DataFrame = self._trafficGenerator.getData(
                        f"{trafficDuration:.0f}s"
                    )

                    trafficData["_time"] = trafficData["_time"] // 1000000000

                    groupedTrafficData: pd.DataFrame = trafficData.groupby(
                        ["_time", "sfcID"]
                    ).agg(
                        reqps=("_value", "count"),
                        medianLatency=("_value", "median"),
                    )

                    latency: float = groupedTrafficData["medianLatency"].mean()

                    self._orchestrator.deleteEmbeddingGraphs(fgs)

                    with open(
                        os.path.join(directory, f"{exp['name']}.txt"),
                        "w",
                        encoding="utf8",
                    ) as f:
                        f.write(
                            f"Time taken: {executionTime:.2f}\n"
                            f"Acceptance Ratio: {acceptanceRatio:.2f}\n"
                            f"Latency: {latency:.2f}\n"
                        )

                    TUI.appendToSolverLog("Finished experiment.")
                    sleep(2)

                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        sfcEm: SFCEmulator = SFCEmulator(SFCRGen, GDASolver, headless)
        sfcEm.startTest(
            topology,
            trafficDesign,
        )
        sfcEm.end()
