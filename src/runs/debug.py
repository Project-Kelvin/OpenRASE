"""
This runs the Simple Dijsktra Algorithm.
"""

import copy
import json
from time import sleep
from timeit import default_timer
from typing import Union
from mano.telemetry import Telemetry
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from shared.models.sfc_request import SFCRequest
from shared.models.config import Config
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from calibrate.calibrate import Calibrate
from mano.orchestrator import Orchestrator
from models.calibrate import ResourceDemand
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile
import click
from utils.tui import TUI
import os

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
    Run the Simple Dijkstra Algorithm.

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
