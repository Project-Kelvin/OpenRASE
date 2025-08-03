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
from utils.traffic_design import generateTrafficDesign

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
    "--headless", default=False, is_flag=True, help="Run in headless mode."
)
def generateData(headless: bool) -> None:
    """
    Generates data for the surrogate model.

    Parameters:
        headless (bool): Run in headless mode.
    """

    print("Generating data for the surrogate model.")
    expName: int = 0
    topology: Topology = None
    # First round
    # trafficDesign: "list[TrafficDesign]" = [
    #     generateTrafficDesign(1, 50, 300)
    # ]

    # Second round
    # trafficDesign: "list[TrafficDesign]" = [generateTrafficDesign(1, 25, 60)]

    # Third round & Fourth round
    # trafficDesign: "list[TrafficDesign]" = [generateTrafficDesign(20, 25, 30)]
    experimentTopologies: "list[tuple[int, Topology, TrafficDesign]]" = [
        ("1_5", generateFatTreeTopology(4, 5, 1, 5120, 1), generateTrafficDesign(1, 50, 300) ),
        ("0.5_20", generateFatTreeTopology(4, 20, 0.5, 5120, 1), generateTrafficDesign(1, 50, 300)),
        ("0.5_5", generateFatTreeTopology(4, 5, 0.5, 5120, 1), generateTrafficDesign(1, 50, 300)),
        ("1_20", generateFatTreeTopology(4, 20, 1, 5120, 1), generateTrafficDesign(1, 50, 300)),
        ("4_2", generateFatTreeTopology(4, 2, 4, 5120, 1), generateTrafficDesign(1, 25, 60)),
        ("0.2_100", generateFatTreeTopology(4, 100, 0.2, 5120, 1), generateTrafficDesign(1, 25, 60)),
        ("0.2_5", generateFatTreeTopology(4, 5, 0.2, 5120, 1), generateTrafficDesign(1, 25, 60)),
        ("4_100", generateFatTreeTopology(4, 100, 4, 5120, 1), generateTrafficDesign(1, 25, 60)),
        ("1_2", generateFatTreeTopology(4, 2, 1, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("0.2_50", generateFatTreeTopology(4, 50, 0.2, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("0.2_5_b", generateFatTreeTopology(4, 5, 0.2, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("1_50", generateFatTreeTopology(4, 50, 1, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("0.5_5_b", generateFatTreeTopology(4, 5, 0.5, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("0.2_20", generateFatTreeTopology(4, 20, 0.2, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("0.5_10", generateFatTreeTopology(4, 10, 0.5, 5120, 1), generateTrafficDesign(20, 25, 30)),
        ("1_20_b", generateFatTreeTopology(4, 20, 1, 5120, 1), generateTrafficDesign(20, 25, 30))
    ]
    # First round
    # lchlExp: tuple[int, Topology] = (1, generateFatTreeTopology(4, 5, 1, 5120, 1))
    # hcllExp: tuple[int, Topology] = (2, generateFatTreeTopology(4, 20, 0.5, 5120, 1))
    # hchlExp: tuple[int, Topology] = (3, generateFatTreeTopology(4, 5, 0.5, 5120, 1))
    # lcllExp: tuple[int, Topology] = (0, generateFatTreeTopology(4, 20, 1, 5120, 1))

    # Second round
    # lchlExp: tuple[int, Topology] = (1, generateFatTreeTopology(4, 2, 4, 5120, 1)) #ignored
    # hcllExp: tuple[int, Topology] = (2, generateFatTreeTopology(4, 100, 0.2, 5120, 1))#ignored
    # hchlExp: tuple[int, Topology] = (3, generateFatTreeTopology(4, 5, 0.2, 5120, 1))#ignored
    # lcllExp: tuple[int, Topology] = (0, generateFatTreeTopology(4, 100, 4, 5120, 1))

    # Third round
    # lchlExp: tuple[int, Topology] = (1, generateFatTreeTopology(4, 2, 1, 5120, 1))#ignored
    # hcllExp: tuple[int, Topology] = (2, generateFatTreeTopology(4, 50, 0.2, 5120, 1))
    # hchlExp: tuple[int, Topology] = (3, generateFatTreeTopology(4, 5, 0.2, 5120, 1))
    # lcllExp: tuple[int, Topology] = (0, generateFatTreeTopology(4, 50, 1, 5120, 1))

    lchlExp: tuple[int, Topology] = (1, generateFatTreeTopology(4, 5, 0.5, 5120, 1))
    hcllExp: tuple[int, Topology] = (2, generateFatTreeTopology(4, 20, 0.2, 5120, 1))
    hchlExp: tuple[int, Topology] = (3, generateFatTreeTopology(4, 10, 0.5, 5120, 1)) #ignored
    lcllExp: tuple[int, Topology] = (0, generateFatTreeTopology(4, 20, 1, 5120, 1))

    for exp in experimentTopologies:
        expName, topology, traffic = exp

        print(f"Generating data for experiment type {expName}.")

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
                        expName
                    )
                    print("Finished experiment.")
                except Exception as e:
                    print(str(e), True)

        trafficDesign: "list[TrafficDesign]" = [traffic]
        sfcEm: SFCEmulator = SFCEmulator(FGR, SFCSolver, headless)
        sfcEm.getSolver().setTrafficDesign(trafficDesign)
        try:
            sfcEm.startTest(topology, trafficDesign)
        except Exception as e:
            print(str(e), True)
        sfcEm.end()
