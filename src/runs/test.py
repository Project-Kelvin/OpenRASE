"""
This file is used to test the functionality of the SFC Emulator.
"""

import os
from timeit import default_timer
import click
import pandas as pd
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from constants.topology import SERVER, SFCC
from mano.telemetry import Telemetry
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.data import hostDataToFrame, mergeHostAndTrafficData
from utils.tui import TUI

artifactsDir: str = os.path.join(getConfig()["repoAbsolutePath"],"artifacts", "test")

if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)

dataPath: str = os.path.join(artifactsDir, "data.csv")

topo: Topology = {
    "hosts": [
        {
            "id": "h1",
            "cpu": 0.7,
            "memory": 1024
        }
    ],
    "switches": [
        {
            "id": "s1"
        }
    ],
    "links": [
        {
            "source": SFCC,
            "destination": "s1",
        },
        {
            "source": "s1",
            "destination": SERVER,
        },
        {
            "source": "h1",
            "destination": "s1",
            "bandwidth": 1000
        }
    ]
}

eg: EmbeddingGraph = {
    "sfcID": "sfc1",
    "sfcrID": "sfcr1",
    "vnfs": {
        "host": {"id": "h1"},
        "vnf": {"id": "waf"},
        "next": {"host": {"id": SERVER}, "next": TERMINAL},
    },
    "links": [
        {"source": {"id": SFCC}, "destination": {"id": "h1"}, "links": ["s1"]},
        {"source": {"id": "h1"}, "destination": {"id": SERVER}, "links": ["s1"]},
    ],
}

trafficDesign: "list[TrafficDesign]" = [
    [
        {
            "target": 2000,
            "duration": "1m"
        }
    ]
]

class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def generateRequests(self) -> None:

        self._orchestrator.sendRequests([eg])

class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def generateEmbeddingGraphs(self) -> None:
        """
        Generate the embedding graphs.
        """

        def updateTUI():
            TUI.appendToSolverLog("Starting traffic generation.")
            duration = 75
            elapsed = 0
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
                trafficData: pd.DataFrame = self._trafficGenerator.getData("75s")
                hostData: pd.DataFrame = hostDataToFrame(hostDataList)
                hostData = mergeHostAndTrafficData(hostData, trafficData)

                hostData.to_csv(dataPath, index=False)
            except Exception as e:
                TUI.appendToSolverLog(f"Error: {e}", True)

            TUI.appendToSolverLog("Solver has finished.")

        self._orchestrator.sendEmbeddingGraphs([eg])
        updateTUI()
        TUI.exit()


@click.command()
@click.option("--headless", default=False, type=bool, is_flag=True, help="Run the emulator in headless mode.")
def run (headless: bool) -> None:
    """
    Run the test.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
    """

    sfcEmulator = SFCEmulator(SFCR, SFCSolver, headless)
    sfcEmulator.startTest(topo, trafficDesign)
    sfcEmulator.startCLI()
    sfcEmulator.end()
