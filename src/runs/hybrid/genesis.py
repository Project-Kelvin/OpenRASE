"""
This runs the Genetic Algorithm
"""

import copy
import json
from time import sleep
from typing import Union

import click
from algorithms.surrogacy.genesis import solve
from mano.orchestrator import Orchestrator
from packages.python.shared.models.config import Config
from packages.python.shared.models.embedding_graph import EmbeddingGraph
from packages.python.shared.models.sfc_request import SFCRequest
from packages.python.shared.models.topology import Topology
from packages.python.shared.models.traffic_design import TrafficDesign
from packages.python.shared.utils.config import getConfig
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI


config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/surrogacy/configs"

topology: Topology = generateFatTreeTopology(4, 5, 1, 5120, 1)

class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self._sfcrs: "list[SFCRequest]" = []

        with open(
            f"{configPath}/sfcrs.json", "r", encoding="utf8"
        ) as sfcrFile:
            sfcrs: "list[SFCRequest]" = json.load(sfcrFile)
            for sfcr in sfcrs:
                self._sfcrs.append(copy.deepcopy(sfcr))

    def generateRequests(self) -> None:
        """
        Generate the requests.
        """

        copiedSFCRs: "list[SFCRequest]" = []
        for index, sfcr in enumerate(self._sfcrs):
            for i in range(0, 8):
                copiedSFCR: SFCRequest = copy.deepcopy(sfcr)
                copiedSFCR["sfcrID"] = f"sfc{index}-{i}"
                copiedSFCRs.append(copiedSFCR)

        self._sfcrs = copiedSFCRs

        self._orchestrator.sendRequests(self._sfcrs)


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def __init__(
        self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator
    ) -> None:
        super().__init__(orchestrator, trafficGenerator)
        self._trafficDesign: "list[TrafficDesign]" = []
        self._trafficType: bool = False
        self._topology: Topology = None

    def setTrafficDesign(self, trafficDesign: "list[TrafficDesign]") -> None:
        """
        Set the traffic design.

        Parameters:
            trafficDesign (list[TrafficDesign]): The traffic design.
        """

        self._trafficDesign: "list[TrafficDesign]" = trafficDesign

    def setTrafficType(self, minimal: bool) -> None:
        """
        Set the traffic type.

        Parameters:
            minimal (bool): Whether to use the minimal traffic design.
        """

        self._trafficType: bool = minimal

    def generateEmbeddingGraphs(self) -> None:
        try:
            while self._requests.empty():
                pass
            requests: "list[Union[EmbeddingGraph, SFCRequest]]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)

            self._topology: Topology = self._orchestrator.getTopology()

            solve(
                requests,
                self._orchestrator.sendEmbeddingGraphs,
                self._orchestrator.deleteEmbeddingGraphs,
                self._trafficDesign,
                self._trafficGenerator,
                self._topology,
                "8_0.1_False_5_1"
            )
            TUI.appendToSolverLog("Finished experiment.")
            sleep(2)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)

        sleep(10)


@click.command()
@click.option(
    "--headless",
    default=False,
    is_flag=True,
    help="If set, the emulator would run in headless mode.",
)
@click.option(
    "--minimal",
    default=False,
    is_flag=True,
    help="If set, the emulator would use the minimal traffic design.",
)
def run(headless: bool, minimal: bool) -> None:
    """
    Run the experiment.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        minimal (bool): Whether to use the minimal traffic design.
    """

    if minimal:
        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesignFromFile(
                f"{getConfig()['repoAbsolutePath']}/src/runs/surrogacy/data/requests.csv",
                0.1,
                1,
                True,
            )
        ]

    else:
        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesignFromFile(
                f"{getConfig()['repoAbsolutePath']}/src/runs/surrogacy/data/requests.csv",
                1,
                4,
            )
        ]
    sfcEm: SFCEmulator = SFCEmulator(SFCR, SFCSolver, headless)
    sfcEm.getSolver().setTrafficDesign(trafficDesign)
    sfcEm.getSolver().setTrafficType(minimal)
    try:
        sfcEm.startTest(topology, trafficDesign)
    except Exception as e:
        TUI.appendToSolverLog(str(e), True)
    sfcEm.end()
