"""
This runs the GAHA algorithm.
"""

import json
import os
from time import sleep
import click
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from algorithms.mak_ga.evolver import gahaEvolve
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile

BANDWIDTH: int = 10  # in MBps
CPUS: int = 1
MEMORY: int = 5120  # in MB
TRAFFIC_SCALE: float = 0.1  # Scale factor for traffic design
TRAFFIC_PATTERN: bool = False
SFCRS: int = 8  # Number of Service Function Chain Requests

topology: Topology = generateFatTreeTopology(4, BANDWIDTH, CPUS, MEMORY)
trafficDesign: list[TrafficDesign] = [
    generateTrafficDesignFromFile(
        os.path.join(
            "src",
            "runs",
            "surrogacy",
            "data",
            "requests.csv"
        ),
        TRAFFIC_SCALE,
        4,
        False,
        TRAFFIC_PATTERN,
    )
]
fgrs: list[EmbeddingGraph] = []

with open(
    os.path.join(
        "src",
        "runs",
        "surrogacy",
        "configs",
        "forwarding-graphs.json"
    ),
    "r",
    encoding="utf8",
) as fgrsFile:
    fgrs = json.load(fgrsFile)


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the GAHA algorithm.

    Parameters:
        headless (bool): If True, run in headless mode.

    Returns:
        None
    """

    class FGR(FGRequestGenerator):
        """
        FGRequestGenerator implementation for GAHA.
        """

        def generateRequests(self) -> None:
            """
            Generate FG requests.

            Returns:
                list[EmbeddingGraph]: The generated FG requests.
            """

            generatedFGRs: list[EmbeddingGraph] = []
            for fi, fgr in enumerate(fgrs):
                for i in range(SFCRS):
                    fgr_copy = fgr.copy()
                    fgr_copy["sfcrID"] = f"sfcr{fi}_{i}"
                    generatedFGRs.append(fgr_copy)

            self._orchestrator.sendFGRequests(generatedFGRs)

    class GAHASolver(Solver):
        """
        Solver implementation for GAHA.
        """

        def generateEmbeddingGraphs(self) -> None:
            """
            Solve the problem using the GAHA algorithm.
            """

            while self._requests.empty():
                pass
            requests: "list[EmbeddingGraph]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)

            gahaEvolve(fgrs, topology, trafficDesign, self._orchestrator.sendEmbeddingGraphs, self._orchestrator.deleteEmbeddingGraphs)

    sfcEmulator: SFCEmulator = SFCEmulator(FGR, GAHASolver, headless)
    sfcEmulator.startTest(
        topology,
        trafficDesign,
    )
    sfcEmulator.end()
