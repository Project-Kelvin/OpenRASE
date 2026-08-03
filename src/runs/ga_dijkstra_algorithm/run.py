"""
This runs the Genetic Algorithm + Dijkstra Algorithm.
"""

import copy
import json
import os
import random
from time import sleep
from typing import Any, Union
import click
import numpy as np
import tensorflow as tf
from algorithms.ga_dijkstra_algorithm.ga import GADijkstraAlgorithm
from mano.orchestrator import Orchestrator
from packages.python.shared.models.config import Config
from packages.python.shared.models.embedding_graph import EmbeddingGraph
from packages.python.shared.models.sfc_request import SFCRequest
from packages.python.shared.models.topology import Topology
from packages.python.shared.models.traffic_design import TrafficDesign
from packages.python.shared.utils.config import getConfig
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI


LINK_BANDWIDTH: int = 10
NO_OF_CPUS: int = 2
MEMORY: int = 5120
TRAFFIC_SCALE: float = 0.1
TRAFFIC_PATTERN: bool = False
NO_OF_COPIES: int = 8

config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/ga_dijkstra_algorithm/configs"

directory = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm"

if not os.path.exists(directory):
    os.makedirs(directory)

logFilePath: str = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/experiments.csv"
)
latencyDataFilePath: str = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/latency_data.csv"
)

@click.command()
@click.option(
    "--headless",
    default=False,
    is_flag=True,
    help="If set, the emulator would run in headless mode.",
)
def run(headless: bool) -> None:
    """
    Run the experiment.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
    """

    experiments: list[tuple[Any]] = [
        (8, 0.1, False, 10, 2), #basic
        (8, 0.1, False, 10, 1), #cpu
        (8, 0.1, False, 5, 2), #bandwidth
        (8, 0.2, False, 10, 2), #traffic scale
        (8, 0.1, True, 10, 2), # traffic pattern
        #(16, 0.1, False, 10, 2),
    ]

    for experiment in experiments:
        topology: Topology = generateFatTreeTopology(
            4, experiment[3], experiment[4], MEMORY, 1
        )

        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesignFromFile(
                os.path.join(
                    f"{getConfig()['repoAbsolutePath']}",
                    "src",
                    "runs",
                    "ga_dijkstra_algorithm",
                    "data",
                    "requests.csv",
                ),
                experiment[1],
                4,
                False,
                experiment[2],
            )
        ]

        class FGR(FGRequestGenerator):
            """
            FG Request Generator.
            """

            def __init__(self, orchestrator: Orchestrator) -> None:
                super().__init__(orchestrator)
                self._fgs: "list[EmbeddingGraph]" = []

                with open(
                    f"{configPath}/forwarding-graphs.json", "r", encoding="utf8"
                ) as fgFile:
                    fgs: "list[EmbeddingGraph]" = json.load(fgFile)
                    for fg in fgs:
                        self._fgs.append(copy.deepcopy(fg))

            def generateRequests(self) -> None:
                """
                Generate the requests.
                """

                copiedFGs: "list[EmbeddingGraph]" = []
                for index, fg in enumerate(self._fgs):
                    for i in range(0, experiment[0]):
                        copiedFG: EmbeddingGraph = copy.deepcopy(fg)
                        copiedFG["sfcrID"] = f"sfc{index}-{i}"
                        copiedFGs.append(copiedFG)

                self._fgs = copiedFGs

                self._orchestrator.sendRequests(self._fgs)

        class SFCSolver(Solver):
            """
            SFC Solver.
            """

            def __init__(
                self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator
            ) -> None:
                super().__init__(orchestrator, trafficGenerator)
                self._topology: Topology = None

            def generateEmbeddingGraphs(self) -> None:
                try:
                    while self._requests.empty():
                        pass
                    requests: "list[Union[FGR, SFCRequest]]" = []
                    while not self._requests.empty():
                        requests.append(self._requests.get())
                        sleep(0.1)
                    self._topology: Topology = self._orchestrator.getTopology()

                    GADijkstraAlgorithm(
                        self._topology,
                        requests,
                        self._orchestrator.sendEmbeddingGraphs,
                        self._orchestrator.deleteEmbeddingGraphs,
                        trafficDesign,
                        self._trafficGenerator,
                        f"{experiment[0]}_{experiment[1]}_{experiment[2]}_{experiment[3]}_{experiment[4]}",
                    )
                    TUI.appendToSolverLog("Finished experiment.")
                    sleep(2)
                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                sleep(10)
                # TUI.exit()

        sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver, headless)
        try:
            sfcEm.startTest(topology, trafficDesign)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)
        sfcEm.end()
