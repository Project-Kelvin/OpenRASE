"""
Defines script to benchmark CPU, memory and link usage against latency.
"""

import os
import random
from time import sleep
import click
import pandas as pd
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.surrogacy.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.vnf_embedding import convertDFtoEGs, convertFGsToDF, getConfidenceValues
from algorithms.surrogacy.utils.scorer import Scorer
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

    global fgs

    class LinkSFCR(SFCRequestGenerator):
        """
        SFC Request Generator.
        """

        def generateRequests(self) -> None:
            egCopy: "list[EmbeddingGraph]" = fgs.copy()
            for i, eg in enumerate(egCopy):
                eg["sfcID"] = f"sfc{i}"
                eg["sfcrID"] = f"sfc{i}"

            self._orchestrator.sendRequests(egCopy)

    class LinksSolver(Solver):
        """
        Class that defines the solver CPU benchmarking.
        """

        def generateEmbeddingGraphs(self):
            """
            This function generates the embedding graphs.
            """

            try:
                while self._requests.empty():
                    pass
                requests: "list[EmbeddingGraph]" = []
                while not self._requests.empty():
                    requests.append(self._requests.get())
                    sleep(0.1)

                finalData: pd.DataFrame = pd.DataFrame()
                for i in range(len(requests)):
                    requestsToDeploy: "list[EmbeddingGraph]" = requests[:i+1]
                    noOfGenes: int = getWeightLength(requests[0], topology)
                    individual: "list[float]" = []
                    eGraphs: "list[EmbeddingGraph]" = []
                    embedData: "dict[str, dict[str, list[click.Tuple[str, int]]]]" = {}
                    weights: "tuple[list[float], list[float], list[float], list[float], list[float], list[float]]" = []
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

                        trafficData["_time"] = trafficData["_time"] // 1000000000

                        groupedTrafficData: pd.DataFrame = trafficData.groupby(
                            ["_time", "sfcID"]
                        ).agg(
                            reqps=("_value", "count"),
                            medianLatency=("_value", "median"),
                        )

                        simulatedReqps: "list[float]" = getTrafficDesignRate(
                            trafficDesign[0],
                            [1]
                            * groupedTrafficData.index.get_level_values(0)
                            .unique()
                            .size,
                        )

                        index: int = 0
                        time: "list[int]" = []
                        sfcIDs: "list[str]" = []
                        realReqps: "list[float]" = []
                        latencies: "list[float]" = []
                        reqps: "list[float]" = []
                        for i, group in groupedTrafficData.groupby(level=0):
                            for eg in finalEGs:
                                time.append(i)
                                sfcIDs.append(eg["sfcID"])
                                realReqps.append(
                                    group.loc[(i, eg["sfcID"])]["reqps"]
                                    if eg["sfcID"] in group.index.get_level_values(1)
                                    else 0
                                )
                                reqps.append(
                                    simulatedReqps[index]
                                    if index < len(simulatedReqps)
                                    else simulatedReqps[-1]
                                )
                                latencies.append(
                                    group.loc[(i, eg["sfcID"])]["medianLatency"]
                                    if eg["sfcID"] in group.index.get_level_values(1)
                                    else 0
                                )
                            index += 1
                        data: pd.DataFrame = pd.DataFrame(
                            {
                                "generation": 1,
                                "individual": 1,
                                "time": time,
                                "sfc": sfcIDs,
                                "reqps": reqps,
                                "real_reqps": realReqps,
                                "latency": latencies,
                                "ar": 1,
                            }
                        )
                        scorer: Scorer = Scorer()
                        data = scorer.getSFCScores(
                            data, topology, finalEGs, embedData, embedLinks.getLinkData()
                        )

                        finalData = pd.concat([finalData, data])
                    except Exception as e:
                        TUI.appendToSolverLog(f"Error: {e}", True)

                    TUI.appendToSolverLog("Solver has finished.")
                    self._orchestrator.deleteEmbeddingGraphs(finalEGs)
                    sleep(5)
                finalData.to_csv(os.path.join(artifactsDir, fileName), index=False)
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)
            TUI.exit()

    sfcEmulator: SFCEmulator = SFCEmulator(LinkSFCR, LinksSolver, headless)
    sfcEmulator.startTest(topology, trafficDesign)
    sfcEmulator.end()
