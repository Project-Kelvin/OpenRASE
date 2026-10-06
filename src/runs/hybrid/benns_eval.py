"""
This file defines the script to run experiments comparing BENNS to GAHA's offline evaluation and online evaluation.
"""

from copy import deepcopy
import os
import random
from time import sleep
import timeit
from typing import cast

import click
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig

from algorithms.hybrid.models.individuals import GenesisIndividual, Individual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from algorithms.mak_ga.mak_ga_utils import MakGAUtils
from algorithms.models.embedding import DecodedIndividual
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateTopologyFromEdgeList
from utils.traffic_design import generateTrafficDesignFromIoTTrace
from utils.tui import TUI

artifactsDir: str = os.path.join(getConfig()["repoAbsolutePath"], "artifacts")
experimentsDir: str = os.path.join(artifactsDir, "experiments")
bennsEvalDir: str = os.path.join(experimentsDir, "benns_eval")
dataFile: str = os.path.join(bennsEvalDir, "data.csv")
if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)
if not os.path.exists(experimentsDir):
    os.makedirs(experimentsDir)
if not os.path.exists(bennsEvalDir):
    os.makedirs(bennsEvalDir)

with open(dataFile, "w") as f:
    f.write("topology,run,egs,ar,benns,gaha,online,benns_time,gaha_time,online_time\n")

@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the BENNS evaluation script.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
    Returns:
        None
    """

    sfcrs: list[SFCRequest] = [
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["waf", "ids", "lb"]
        }),
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["ips", "ids", "tm"]
        }),
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["ips", "ha", "ids"]
        }),
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["lb", "ids"]
        })
    ]

    noOfCopies: int = 8
    noOfRuns: int = 100

    abilene: Topology = generateTopologyFromEdgeList(os.path.join(
        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "abilene.txt"),
        1,
        5 * 1024,
        10,
        1
    )
    dfnBwin: Topology = generateTopologyFromEdgeList(os.path.join(
        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "dfn-bwin.txt"),
        1,
        5 * 1024,
        10,
        1
    )
    nobelUs: Topology = generateTopologyFromEdgeList(os.path.join(
        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "nobel-us.txt"),
        1,
        5 * 1024,
        10,
        1
    )

    topologies: list[Topology] = [abilene, dfnBwin, nobelUs]

    design: TrafficDesign = generateTrafficDesignFromIoTTrace(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "hybrid",
            "data",
            "iot-trace-2.csv",
        ),
        2,
        750,
    )


    for topoIndex, topology in enumerate(topologies):

        class SFCRGen(SFCRequestGenerator):
            """
            Class to generate FG Requests.
            """

            def generateRequests(self) -> None:
                """
                Generate the FG Requests.
                """

                sfcrsToSend: list[SFCRequest] = []

                for i, sfcr in enumerate(sfcrs):
                    for copy in range(noOfCopies):
                        sfcrToSend: SFCRequest = deepcopy(sfcr)
                        sfcrToSend["sfcrID"] = f"sfcr{i}_{copy}"
                        sfcrsToSend.append(sfcrToSend)

                self._orchestrator.sendRequests(sfcrsToSend)

        class EvalSolver(Solver):
            """
            Class to run the BENNS evaluation.
            """

            def generateEmbeddingGraphs(self):
                """
                Generate the embedding graphs.
                """

                try:
                    while self._requests.empty():
                        pass
                    requests: "list[SFCRequest]" = []
                    while not self._requests.empty():
                        requests.append(self._requests.get())
                        sleep(0.1)

                    topologyName: str = "abilene"

                    if topoIndex == 1:
                        topologyName = "dfn-bwin"
                    elif topoIndex == 2:
                        topologyName = "nobel-us"

                    for run in range(noOfRuns):
                        noOfSFCRsToEmbed: int = random.randint(15, len(requests))
                        sfcrsToEmbed: list[SFCRequest] = random.sample(requests, noOfSFCRsToEmbed)
                        print(f"Embedding graphs: {len(sfcrsToEmbed)}")
                        print(f"Running BENNS evaluation for topology {topologyName} run {run + 1}/{noOfRuns}...")

                        GenesisUtils.init(sfcrsToEmbed, topology, 6, 0.0, 1)
                        ind: Individual = GenesisUtils.generateRandomGenesisIndividual(Individual, topology, sfcrsToEmbed)
                        decodedInd: list[DecodedIndividual] = [GenesisUtils.decodeIndividual(cast(GenesisIndividual, ind), 0, topology, sfcrsToEmbed)]

                        makGAUtils = MakGAUtils(topology, design, sfcrsToEmbed)
                        makGAUtils.cacheDemand(decodedInd)
                        HybridEvaluation.cacheForOffline(
                            decodedInd, [design], topology, 0, isAvgOnly=True
                        )

                        bennsStartTime: float = timeit.default_timer()
                        bennsEval: tuple[int, float, float] = HybridEvaluation.evaluationOnSurrogate(decodedInd[0], 0, 1, topology, [design], 1000000000000)
                        bennsEndTime: float = timeit.default_timer()
                        bennsTime: float = round(bennsEndTime - bennsStartTime, 2)

                        gahaStartTime: float = timeit.default_timer()
                        gahaEval: tuple[int, float, float] = makGAUtils.getTotalDelay(decodedInd[0])
                        gahaEndTime: float = timeit.default_timer()
                        gahaTime: float = round(gahaEndTime - gahaStartTime, 2)

                        openRaseStartTime: float = timeit.default_timer()
                        openRaseEval: tuple[float, float] = HybridEvaluation.evaluationOnEmulator(decodedInd[0], sfcrsToEmbed, 0, 1, self._orchestrator.sendEmbeddingGraphs, self._orchestrator.deleteEmbeddingGraphs, [design], self._trafficGenerator,topology, 1000000000000)
                        openRaseEndTime: float = timeit.default_timer()
                        openRaseTime: float = round(openRaseEndTime - openRaseStartTime, 2)

                        with open(dataFile, "a") as f:
                            f.write(f"{topologyName},{run + 1},{len(decodedInd[0][1])},{openRaseEval[0]},{bennsEval[2]},{gahaEval[2]},{openRaseEval[1]},{bennsTime},{gahaTime},{openRaseTime}\n")
                except Exception as e:
                    TUI.appendToSolverLog(f"Error while generating embedding graphs: {e}")

        sfcEm: SFCEmulator = SFCEmulator(SFCRGen, EvalSolver, headless)
        sfcEm.startTest(
            topology,
            [design],
        )
        sfcEm.end()
