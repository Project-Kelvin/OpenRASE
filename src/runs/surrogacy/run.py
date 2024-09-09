"""
This runs the Genetic Algorithm + Dijsktra Algorithm.
"""

import copy
import json
import os
from time import sleep
from typing import Union

import click
from algorithms.surrogacy.ga import evolveWeights
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
from utils.tui import TUI


config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/ga_dijkstra_algorithm/configs"


directory = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm"

if not os.path.exists(directory):
    os.makedirs(directory)

topology: Topology = generateFatTreeTopology(4, 1000, 2, None)
trafficDesign: "list[TrafficDesign]" = []
logFilePath: str = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/experiments.csv"
latencyDataFilePath: str = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/latency_data.csv"

def appendToLog(message: str) -> None:
    """
    Append to the log.

    Parameters:
        message (str): The message to append.
    """

    with open(logFilePath, "a", encoding="utf8") as log:
        log.write(f"{message}\n")

with open(f"{configPath}/traffic-design.json", "r", encoding="utf8") as trafficFile:
    trafficDesign = [json.load(trafficFile)]

class FGR(FGRequestGenerator):
    """
    FG Request Generator.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self._fgs: "list[EmbeddingGraph]" = []

        with open(f"{configPath}/forwarding-graphs.json", "r", encoding="utf8") as fgFile:
            fgs: "list[EmbeddingGraph]" = json.load(fgFile)
            for fg in fgs:
                self._fgs.append(copy.deepcopy(fg))

    def generateRequests(self) -> None:
        """
        Generate the requests.
        """

        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            for i in range(0, 8):
                copiedFG: EmbeddingGraph = copy.deepcopy(fg)
                copiedFG["sfcrID"] = f"sfc{index}-{i}"
                copiedFGs.append(copiedFG)

        self._fgs = copiedFGs

        self._orchestrator.sendRequests(self._fgs)

class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__(orchestrator, trafficGenerator)
        self._topology: Topology = self._orchestrator.getTopology()

    def generateEmbeddingGraphs(self) -> None:
        try:
            while self._requests.empty():
                pass
            requests: "list[Union[FGR, SFCRequest]]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)

            evolveWeights(requests, self._orchestrator.sendEmbeddingGraphs, self._orchestrator.deleteEmbeddingGraphs, trafficDesign, self._trafficGenerator, self._topology)
            TUI.appendToSolverLog(f"Finished experiment.")
            sleep(2)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)

        sleep(10)
        #TUI.exit()

@click.command()
@click.option("--headless", default=False, is_flag=True, help="If set, the emulator would run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the experiment.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
    """

    sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver, headless)
    try:
        sfcEm.startTest(topology, trafficDesign)
    except Exception as e:
        TUI.appendToSolverLog(str(e), True)
    sfcEm.end()
