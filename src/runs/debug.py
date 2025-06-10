"""
This runs the SFC Emulator to debug it.
"""

import json
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.models.topology import Topology
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology


config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"

topology: Topology = generateFatTreeTopology(4, 1000, 2, None)
trafficDesignPath = f"{configPath}/traffic-design.json"
with open(trafficDesignPath, "r", encoding="utf8") as traffic:
    trafficDesign = json.load(traffic)

class FGR(FGRequestGenerator):
    """
    FG Request Generator.
    """

    def generateRequests(self):
        while True:
            pass


def run() -> None:
    """
    Runs the Emulator to debug it.

    Parameters:
        experiment (int): The experiment to run.
    """
    sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver)
    sfcEm.startTest(topology, trafficDesign)
    sfcEm.end()


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def generateEmbeddingGraphs(self) -> None:
        while True:
            pass
