"""
This trains the surrogate model.
"""

import copy
import json
import os
from time import sleep
from typing import Union
import click
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.surrogacy.constants.surrogate import SURROGACY_PATH
from algorithms.surrogacy.surrogate.data_generator.evolver import evolveWeights
from algorithms.surrogacy.surrogate.surrogate import train
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI

config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/surrogacy/configs"

directory = SURROGACY_PATH

if not os.path.exists(directory):
    os.makedirs(directory)


def trainModel() -> None:
    """
    Runs the surrogate model.
    """

    train()


@click.command()
@click.option(
    "--lcll", default=False, is_flag=True, help="Low CPU usage. Low link usage."
)
@click.option(
    "--lchl", default=False, is_flag=True, help="Low CPU usage. High link usage."
)
@click.option(
    "--hcll", default=False, is_flag=True, help="High CPU usage. Low link usage."
)
@click.option(
    "--hchl", default=False, is_flag=True, help="High CPU usage. High link usage."
)
# pylint: disable=unused-argument
def generateData(lcll: bool, lchl: bool, hcll: bool, hchl: bool) -> None:
    """
    Generates data for the surrogate model.

    Parameters:
        lcll (bool): Low CPU usage. Low link usage.
        lchl (bool): Low CPU usage. High link usage.
        hcll (bool): High CPU usage. Low link usage.
        hchl (bool): High CPU usage. High link usage.
    """

    dataType: int = 0
    topology: Topology = None

    if lchl:
        dataType = 1
        topology = generateFatTreeTopology(4, 3, 1, 5120)
    elif hcll:
        dataType = 2
        topology = generateFatTreeTopology(4, 6, 0.2, 5120)
    elif hchl:
        dataType = 3
        topology = generateFatTreeTopology(4, 3, 0.2, 5120)
    else:
        dataType = 0
        topology = generateFatTreeTopology(4, 6, 1, 5120)

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
                for fg in fgs[:2]:
                    self._fgs.append(copy.deepcopy(fg))

        def generateRequests(self) -> None:
            """
            Generate the requests.
            """

            copiedFGs: "list[EmbeddingGraph]" = []
            for index, fg in enumerate(self._fgs):
                for i in range(0, 16):
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
            self._trafficDesign: "list[TrafficDesign]" = []
            self._topology: Topology = None

        def setTrafficDesign(self, trafficDesign: "list[TrafficDesign]") -> None:
            """
            Set the traffic design.

            Parameters:
                trafficDesign (list[TrafficDesign]): The traffic design.
            """

            self._trafficDesign: "list[TrafficDesign]" = trafficDesign

        def generateEmbeddingGraphs(self) -> None:
            self._topology = self._orchestrator.getTopology()
            try:
                while self._requests.empty():
                    pass
                requests: "list[Union[EmbeddingGraph, SFCRequest]]" = []
                while not self._requests.empty():
                    requests.append(self._requests.get())
                    sleep(0.1)

                evolveWeights(
                    requests,
                    self._orchestrator.sendEmbeddingGraphs,
                    self._orchestrator.deleteEmbeddingGraphs,
                    self._trafficDesign,
                    self._trafficGenerator,
                    self._topology,
                    dataType
                )
                TUI.appendToSolverLog("Finished experiment.")
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)

    trafficDesign: "list[TrafficDesign]" = [
        generateTrafficDesignFromFile(
            f"{getConfig()['repoAbsolutePath']}/src/runs/surrogacy/data/requests.csv",
            0.1,
            1,
            True,
        )
    ]

    sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver, True)
    sfcEm.getSolver().setTrafficDesign(trafficDesign)
    try:
        sfcEm.startTest(topology, trafficDesign)
    except Exception as e:
        TUI.appendToSolverLog(str(e), True)
    sfcEm.end()
