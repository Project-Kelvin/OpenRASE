"""
Defines script to benchmark CPU, memory and link usage against latency.
"""

import os
import random
from time import sleep
import click
import pandas as pd
import polars as pl
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.models.traffic import TimeSFCRequests
from algorithms.surrogacy.utils.demand_predictions import DemandPredictions
from algorithms.surrogacy.utils.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.solvers.link_embedding import EmbedLinks
from algorithms.surrogacy.solvers.vnf_embedding import convertDFtoEGs, convertFGsToDF, getConfidenceValues
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesign, getTrafficDesignRate
from utils.tui import TUI

fgs: "list[EmbeddingGraph]" = [
    {
        "vnfs": {
            "vnf": {"id": "lb"},
            "next": [
                {
                    "vnf": {"id": "waf"},
                    "next": {"host": {"id": "server"}, "next": "terminal"},
                },
                {
                    "vnf": {"id": "waf"},
                    "next": {"host": {"id": "server"}, "next": "terminal"},
                },
            ],
        }
    },
]

trafficDuration: int = 300

artifactsDir: str = os.path.join(
    getConfig()["repoAbsolutePath"], "artifacts", "experiments", "surrogacy"
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
# pylint: disable=unused-argument
def benchmark(headless: bool, control: bool, cpu: bool, memory: bool, links: bool) -> None:
    """
    This function benchmarks link usage against latency.

    Parameters:
        headless (bool): Run headless.
        control (bool): Run control.
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

    if cpu:
        cpus = 0.5
        fileName = cpuDataPath
    elif memory:
        memoryAmount = 512
        fileName = memoryDataPath
    elif links:
        bandwidth = 3
        fileName = linksDataPath
    else:
        fileName = controlDataPath

    topology: Topology = generateFatTreeTopology(4, bandwidth, cpus, memoryAmount)
    trafficDesign: "list[TrafficDesign]" = [
        generateTrafficDesign(0, 600, trafficDuration)
    ]

    class ToySFCR(SFCRequestGenerator):
        """
        SFC Request Generator.
        """

        def generateRequests(self) -> None:
            egCopy: "list[EmbeddingGraph]" = fgs.copy()
            for i, eg in enumerate(egCopy):
                eg["sfcID"] = f"sfc{i}"
                eg["sfcrID"] = f"sfc{i}"

            self._orchestrator.sendRequests(egCopy)

    class ToySolver(Solver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            """
            This function generates the embedding graphs.
            """

            demandPredictions: DemandPredictions = DemandPredictions()

            try:
                while self._requests.empty():
                    pass
                requests: "list[EmbeddingGraph]" = []
                while not self._requests.empty():
                    requests.append(self._requests.get())
                    sleep(0.1)

                finalData: pl.DataFrame = None
                for i in range(len(requests)):
                    requestsToDeploy: "list[EmbeddingGraph]" = requests[:i+1]
                    noOfGenes: int = getWeightLength(requests[0], topology)
                    individual: "list[float]" = []
                    eGraphs: "list[EmbeddingGraph]" = []
                    embedData: "dict[str, dict[str, list[click.Tuple[str, int]]]]" = {}
                    weights: "tuple[list[float], list[float], list[float], list[float], list[float], list[float]]" = []
                    hybridEvolution: HybridEvolution = HybridEvolution()
                    while len(eGraphs) != len(requestsToDeploy):
                        for _ in range(noOfGenes):
                            individual.append(random.uniform(-1, 1))
                        weights = getWeights(individual, requestsToDeploy, topology)
                        eGraphs, nodes, embedData = convertDFtoEGs(
                            getConfidenceValues(
                                convertFGsToDF(requestsToDeploy, topology),
                                weights[2],
                                weights[3],
                            ),
                            requestsToDeploy,
                            topology,
                        )

                    embedLinks: EmbedLinks = EmbedLinks(
                        topology, requestsToDeploy, weights[4], weights[5]
                    )
                    finalEGs: "list[EmbeddingGraph]" = embedLinks.embedLinks(nodes)

                    self._orchestrator.sendEmbeddingGraphs(finalEGs)
                    TUI.appendToSolverLog("Starting traffic generation.")
                    sleep(trafficDuration)
                    try:
                        trafficData: pd.DataFrame = self._trafficGenerator.getData(
                            f"{trafficDuration}s"
                        )

                        ar: int = len(finalEGs) / len(requestsToDeploy)
                        decodedIndividual: DecodedIndividual = (0, finalEGs, embedData, embedLinks.getLinkData(), ar)
                        hybridEvolution.cacheForOffline([decodedIndividual], trafficDesign, topology, 1)
                        data: pl.DataFrame = hybridEvolution.generateScoresForRealTrafficData(
                            decodedIndividual,
                            trafficData,
                            trafficDesign,
                            topology
                        )
                        finalData = pd.concat([finalData, data]) if finalData is not None else data
                    except Exception as e:
                        TUI.appendToSolverLog(f"Error: {e}", True)

                    TUI.appendToSolverLog("Solver has finished.")
                    self._orchestrator.deleteEmbeddingGraphs(finalEGs)
                    sleep(5)
                finalData.write_csv(os.path.join(artifactsDir, fileName))
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)
            TUI.exit()

    sfcEmulator: SFCEmulator = SFCEmulator(ToySFCR, ToySolver, headless)
    sfcEmulator.startTest(topology, trafficDesign)
    sfcEmulator.end()
