"""
This runs the Simple Dijsktra Algorithm.
"""

import copy
import json
from shared.models.config import Config
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from calibrate.calibrate import Calibrate
from mano.orchestrator import Orchestrator
from models.calibrate import ResourceDemand
from packages.shared.models.embedding_graph import EmbeddingGraph
from packages.shared.models.topology import Topology
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesign


config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"

topology: Topology = generateFatTreeTopology(4, 1000, 1, None)

with open(f"{configPath}/traffic-design.json", "r", encoding="utf8") as trafficFile:
    trafficDesign: "list[TrafficDesign]" = [json.load(trafficFile)]

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
                for _i in range (2):
                    self._fgs.append(copy.deepcopy(fg))


    def generateRequests(self) -> None:
        for index, fg in enumerate(self._fgs):
            fg["sfcID"] = f"sfc{index}"

        self._orchestrator.sendRequests(self._fgs)

class SFCSolver(Solver):
    """
    SFC Solver.
    """


    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__(orchestrator, trafficGenerator)
        self._topology: Topology = None
        self._resourceDemands: "dict[str, ResourceDemand]" = None

        calibrate = Calibrate()

        trafficDesignPath = f"{configPath}/traffic-design.json"
        with open(trafficDesignPath, "r", encoding="utf8") as traffic:
            design = json.load(traffic)
        maxTarget: int = max(design, key=lambda x: x["target"])["target"]

        self._resourceDemands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(maxTarget)

    def generateEmbeddingGraphs(self) -> None:
        sda = SimpleDijkstraAlgorithm(self._requests, topology, self._resourceDemands)
        fgs, failedFGs, _nodes = sda.run()
        print(f"Failed to deploy {len(failedFGs)} out of {len(fgs) + len(failedFGs)} FGs.")
        self._orchestrator.sendEmbeddingGraphs(fgs)

def run() -> None:
    """
    Run the Simple Dijkstra Algorithm.
    """

    sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver)
    sfcEm.startTest(topology, trafficDesign)
    sfcEm.startCLI()
    sfcEm.end()

def getTrafficDesign() -> None:
    """
    Get the Traffic Design.
    """

    design: TrafficDesign = generateTrafficDesign(
        f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/requests.csv")

    with open(f"{configPath}/traffic-design.json", "w", encoding="utf8") as traffic:
        json.dump(design, traffic, indent=4)
