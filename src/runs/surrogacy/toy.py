"""
Defines script to benchmark CPU, memory and link usage against latency.
"""

import os
from time import sleep
import click
import pandas as pd
import polars as pl
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.ga_dijkstra_algorithm.ga_utils import generateRandomIndividual, convertIndividualToEmbeddingGraph
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesign
from utils.tui import TUI
import json

NUM_OF_ROUNDS: int = 5

fgs_path = os.path.join(getConfig()["repoAbsolutePath"],"src","runs","surrogacy","configs","forwarding-graphs.json")
with open(fgs_path, "r") as f:
    fgs: "list[EmbeddingGraph]" = json.load(f)
    fgs = fgs[0:2]

artifactsDir: str = os.path.join(
    getConfig()["repoAbsolutePath"], "artifacts", "experiments", "toy"
)

if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)

cpuDataPath: str = os.path.join(artifactsDir, "cpu.csv")
memoryDataPath: str = os.path.join(artifactsDir, "memory.csv")
controlDataPath: str = os.path.join(artifactsDir, "control.csv")
linksDataPath: str = os.path.join(artifactsDir, "links.csv")


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run headless.")
@click.option("--control", is_flag=True, default=True, help="Run control.")
@click.option("--cpu", is_flag=True, default=False, help="Run CPU benchmark.")
@click.option("--memory", is_flag=True, default=False, help="Run memory benchmark.")
@click.option("--links", is_flag=True, default=False, help="Run link benchmark.")
@click.option("--all", is_flag=True, default=False, help="Run all benchmarks.")
# pylint: disable=unused-argument
def benchmark(headless: bool, control: bool, cpu: bool, memory: bool, links: bool, all: bool) -> None:
    """
    This function benchmarks link usage against latency.

    Parameters:
        headless (bool): Run headless.
        control (bool): Run control.
        cpu (bool): Run CPU benchmark.
        memory (bool): Run memory benchmark.
        links (bool): Run link benchmark.
        all (bool): Run all benchmarks.

    Returns:
        None
    """

    def runToy(cpu: bool, memory: bool, links: bool) -> None:
        """
        This function runs the toy benchmark.

        Parameters:
            cpu (bool): Run CPU benchmark.
            memory (bool): Run memory benchmark.
            links (bool): Run link benchmark.

        Returns:
            None
        """

        cpus: float = None
        memoryAmount: int = None
        bandwidth: int = None
        fileName: str = None

        trafficDuration: int = 120  # seconds

        if cpu:
            cpus = 0.5
            fileName = cpuDataPath
        elif memory:
            memoryAmount = 1024
            fileName = memoryDataPath
        elif links:
            bandwidth = 4
            fileName = linksDataPath
        else:
            fileName = controlDataPath

        topology: Topology = generateFatTreeTopology(4, bandwidth, cpus, memoryAmount, 1)
        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesign(0, 250, trafficDuration)
        ]

        class ToySFCR(SFCRequestGenerator):
            """
            SFC Request Generator.
            """

            def generateRequests(self) -> None:
                egCopy: "list[EmbeddingGraph]" = fgs.copy()
                egs: "list[EmbeddingGraph]" = []
                for i, eg in enumerate(egCopy):
                    for copy in range(0, 1):
                        eg["sfcID"] = f"sfc{i}_{copy}"
                        eg["sfcrID"] = f"sfc{i}_{copy}"
                        egs.append(eg.copy())

                self._orchestrator.sendRequests(egs)

        class ToySolver(Solver):
            """
            Class that defines the solver CPU benchmarking.
            """

            def generateEmbeddingGraphs(self):
                """
                This function generates the embedding graphs.
                """

                class Container(list):
                    def __init__(self):
                        self.id = 1
                try:
                    while self._requests.empty():
                        pass
                    requests: "list[EmbeddingGraph]" = []
                    while not self._requests.empty():
                        requests.append(self._requests.get())
                        sleep(0.1)

                    finalData: pl.DataFrame = None

                    for r in range(NUM_OF_ROUNDS):
                        requestsToDeploy: "list[EmbeddingGraph]" = requests
                        eGraphs: "list[EmbeddingGraph]" = []
                        hybridEvolution: HybridEvolution = HybridEvolution()
                        while len(eGraphs) != len(requestsToDeploy):
                            individual: list[list[float]] = generateRandomIndividual(Container, topology, requestsToDeploy, 0.01)
                            if not links:
                                for i in range(len(individual)):
                                    individual[i] = [0] * len(individual[i])
                                    individual[i][0] = 1
                            output = convertIndividualToEmbeddingGraph(individual, requestsToDeploy, topology)
                            eGraphs = output[0]
                            TUI.appendToSolverLog(
                                f"Acceptance Ratio: {len(eGraphs) / len(requestsToDeploy)}"
                            )

                        self._orchestrator.sendEmbeddingGraphs(eGraphs)
                        TUI.appendToSolverLog("Starting traffic generation.")
                        sleep(trafficDuration)
                        try:
                            trafficData: pd.DataFrame = self._trafficGenerator.getData(
                                f"{trafficDuration}s"
                            )

                            ar: int = len(eGraphs) / len(requestsToDeploy)
                            decodedIndividual: DecodedIndividual = (0, eGraphs, output[1], output[2], ar)
                            hybridEvolution.cacheForOffline([decodedIndividual], trafficDesign, topology, 1)
                            data: pl.DataFrame = hybridEvolution.generateScoresForRealTrafficData(
                                decodedIndividual,
                                trafficData,
                                trafficDesign,
                                topology,
                                r
                            )
                            finalData = pl.concat([finalData, data]) if finalData is not None else data
                        except Exception as e:
                            TUI.appendToSolverLog(f"Error: {e}", True)

                        TUI.appendToSolverLog("Solver has finished.")
                        self._orchestrator.deleteEmbeddingGraphs(eGraphs)
                        sleep(60)
                    finalData.write_csv(os.path.join(artifactsDir, fileName))
                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Solver has finished all requests.")
                TUI.exit()

        sfcEmulator: SFCEmulator = SFCEmulator(ToySFCR, ToySolver, headless)
        sfcEmulator.startTest(topology, trafficDesign)
        sfcEmulator.end()

    if all:
        runToy(True, False, False)
        runToy(False, True, False)
        runToy(False, False, True)
        #runToy(False, False, False)
    elif cpu:
        runToy(True, False, False)
    elif memory:
        runToy(False, True, False)
    elif links:
        runToy(False, False, True)
    else:
        runToy(False, False, False)
