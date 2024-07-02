"""
This runs the Genetic Algorithm + Dijsktra Algorithm.
"""

import copy
import json
import os
from time import sleep
from typing import Union
from algorithms.ga_dijkstra_algorithm.utils import convertIndividualToEmbeddingGraph, generateRandomIndividual, getVNFsfromFGRs, validateIndividual
from calibrate.calibrate import Calibrate
from mano.orchestrator import Orchestrator
from models.calibrate import ResourceDemand
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
from utils.traffic_design import generateTrafficDesign
from utils.tui import TUI


config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/ga_dijkstra_algorithm/configs"


directory = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm"

if not os.path.exists(directory):
    os.makedirs(directory)

topology: Topology = generateFatTreeTopology(4, 1000, 2, None)
logFilePath: str = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/experiments.csv"
latencyDataFilePath: str = f"{config['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/latency_data.csv"

with open(logFilePath, "w", encoding="utf8") as log:
    log.write("experiment,failed_fgs,accepted_fgs,execution_time,deployment_time\n")

with open(latencyDataFilePath, "w", encoding="utf8") as latencyData:
    latencyData.write("experiment,sfc,requests,average_latency,duration\n")

def appendToLog(message: str) -> None:
    """
    Append to the log.

    Parameters:
        message (str): The message to append.
    """

    with open(logFilePath, "a", encoding="utf8") as log:
        log.write(f"{message}\n")

with open(f"{configPath}/traffic-design.json", "r", encoding="utf8") as trafficFile:
    trafficDesign: "list[TrafficDesign]" = [json.load(trafficFile)]


def getFGRs() -> "list[EmbeddingGraph]":
    """
    Get the FG Request.

    Returns:
        list[EmbeddingGraph]: the FG Request.
    """

    fgs: "list[EmbeddingGraph]" = []
    with open(f"{configPath}/forwarding-graphs.json", "r", encoding="utf8") as fgFile:
            fgs = json.load(fgFile)

    return fgs

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
        self._resourceDemands: "dict[str, ResourceDemand]" = None

        calibrate = Calibrate()

        trafficDesignPath = f"{configPath}/traffic-design.json"
        with open(trafficDesignPath, "r", encoding="utf8") as traffic:
            design = json.load(traffic)
        maxTarget: int = max(design, key=lambda x: x["target"])["target"]

        self._resourceDemands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(maxTarget)

    def generateEmbeddingGraphs(self) -> None:
        try:
            while self._requests.empty():
                pass
            requests: "list[Union[FGR, SFCRequest]]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)
            self._topology: Topology = self._orchestrator.getTopology()


            TUI.appendToSolverLog(f"Finished experiment.")
            sleep(2)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)
        TUI.exit()

def getTrafficDesign() -> None:
    """
    Get the Traffic Design.
    """

    design: TrafficDesign = generateTrafficDesign(
        f"{getConfig()['repoAbsolutePath']}/src/runs/ga_dijkstra_algorithm/data/requests.csv", 2)

    with open(f"{configPath}/traffic-design.json", "w", encoding="utf8") as traffic:
        json.dump(design, traffic, indent=4)

def run() -> None:
    """
    Run the experiment
    """

    sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver)
    sfcEm.startTest(topology, trafficDesign)
    sfcEm.end()

def test() -> None:
    """
    Tests
    """

    trafficDesignPath = f"{configPath}/traffic-design.json"
    with open(trafficDesignPath, "r", encoding="utf8") as traffic:
        design = json.load(traffic)
    maxTarget: int = max(design, key=lambda x: x["target"])["target"]
    calibrate: Calibrate = Calibrate()
    resourceDemands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(maxTarget)
    ind: "list[list[int]]" = generateRandomIndividual(14, topology, resourceDemands, getFGRs())
    sampleInvalidInd: "list[list[int]]" = [
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ]
    sampleValidInd: "list[list[int]]" = [
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    ]
    limitedTopo: Topology = generateFatTreeTopology(4, 1000, 0.5, None)
    print(getVNFsfromFGRs(getFGRs()))
    print(ind)
    print(validateIndividual(sampleInvalidInd, limitedTopo, resourceDemands, getFGRs()))
    print(validateIndividual(sampleValidInd, limitedTopo, resourceDemands, getFGRs()))
    print(convertIndividualToEmbeddingGraph(ind, getFGRs()))
