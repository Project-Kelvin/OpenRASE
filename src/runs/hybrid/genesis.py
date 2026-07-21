"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import json
import os
from time import sleep
from typing import Any
import click
import numpy as np
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.genesis import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI

MUT_PB: float = 0.7 # Experimentally determined mutation probability for the GA
GENE_MUT_PB: float = 0.7 # Experimentally determined gene mutation probability for the GA
CX_PB: float = 1.0

@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--mutation", is_flag=True, default=False, help="Run in mutation pbs hyperparameter tuning mode.")
@click.option("--cx", is_flag=True, default=False, help="Run in crossover pb hyperparameter tuning mode.")
@click.option("--rr", is_flag=True, default=False, help="Run in Rejection Rate tuning mode.")
@click.option("--sigma", is_flag=True, default=False, help="Run in sigma hyperparameter tuning mode.")
@click.option("--chain", is_flag=True, default=False, help="Use static chain decoding.")
@click.option("--dijkstra", is_flag=True, default=False, help="Use Dijkstra's algorithm for pathfinding.")
@click.option("--gaussian", is_flag=True, default=True, help="Disable the Gaussian distribution for host selection.")
@click.option("--activation", is_flag=True, default=False, help="Test activation functions in the neural network.")
@click.option("--init", is_flag=True, default=False, help="Test the limit to use for generating the predefined weights.")
def run(headless: bool, mutation: bool, cx: bool, rr: bool, sigma: bool, chain: bool, dijkstra: bool, gaussian: bool, activation: str, init: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        mutation (bool): Whether to run in mutation probability hyperparameter tuning mode.
        cx (bool): Whether to run in crossover probability hyperparameter tuning mode.
        rr (bool): Whether to run in Rejection Rate tuning mode.
        sigma (bool): Whether to run in sigma hyperparameter tuning mode.
        chain (bool): Whether to use static chain decoding.
        dijkstra (bool): Whether to use Dijkstra's algorithm for pathfinding.
        gaussian (bool): Whether to disable the Gaussian distribution for host selection.
        activation (str): Whether to test activation functions in the neural network.
        init (bool): Whether to test the limit to use for generating the predefined weights.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    rejectionRates: list[float] = [0.05, 0.07, 0.1]
    sigmas: list[float] = [0.0, 1.0, 2.0, 4.0]
    activations: list[str] = ["sin", "tanh", "relu", "linear"]
    initLimit: list[float] = [1, 2, np.pi, 2 * np.pi]

    experimentsIncludeFilter: list[tuple[int, float, bool, int, int]] = [
        (20, 0.1, False, 10, 1), # Hard
        (12, 0.1, False, 10, 2), # Medium
        (8, 0.1, False, 10, 2), # Easy
    ]

    if mutation or cx or rr or sigma or activation or init or chain or dijkstra or gaussian:
        experimentsIncludeFilter = [experimentsIncludeFilter[0]]  # Only run the Hard experiment for hyperparameter tuning

    noOfRuns: int = 20

    experimentsExcludeFilter: list[tuple[int, float, bool, int, int]] = []
    experimentPriority: list[str] = []
    experimentsToRun: list[dict[str, Any]] = []

    for noOfCopy in [20, 12, 8]:
        for trafficScale in [0.1, 0.2]:
            for trafficPattern in [False, True]:
                for linkBandwidth in [10, 5]:
                    for noOfCPUs in [2, 1, 0.5]:
                        experimentsToRun.append(
                            {
                                "name": f"{noOfCopy}_{trafficScale}_{trafficPattern}_{linkBandwidth}_{noOfCPUs}",
                                "noOfCopies": noOfCopy,
                                "trafficScale": trafficScale,
                                "trafficPattern": trafficPattern,
                                "linkBandwidth": linkBandwidth,
                                "noOfCPUs": noOfCPUs,
                                "memory": 5120,
                            }
                        )

    if len(experimentPriority) > 0:
        experimentsToRun = sorted(
            experimentsToRun,
            key=lambda x: (
                experimentPriority.index(x["name"])
                if x["name"] in experimentPriority
                else len(experimentPriority)
            ),
        )

    for exp in experimentsToRun:
        if (
            len(experimentsIncludeFilter) > 0
            and (
                exp["noOfCopies"],
                exp["trafficScale"],
                exp["trafficPattern"],
                exp["linkBandwidth"],
                exp["noOfCPUs"],
            )
            not in experimentsIncludeFilter
        ):
            continue

        if (
            len(experimentsExcludeFilter) > 0
            and (
                exp["noOfCopies"],
                exp["trafficScale"],
                exp["trafficPattern"],
                exp["linkBandwidth"],
                exp["noOfCPUs"],
            )
            in experimentsExcludeFilter
        ):
            continue

        class SFCRGen(SFCRequestGenerator):
            """
            Class to generate FG Requests.
            """

            def __init__(self, orchestrator: Orchestrator) -> None:
                """
                Initialize the SFCRGen class.
                """

                super().__init__(orchestrator)
                with open(
                    os.path.join(
                        getConfig()["repoAbsolutePath"],
                        "src",
                        "runs",
                        "hybrid",
                        "configs",
                        "sfcrs.json",
                    ),
                    "r",
                    encoding="utf8",
                ) as f:
                    self.sfcrs = json.load(f)

            def generateRequests(self) -> "list[EmbeddingGraph]":
                """
                Generate the FG Requests.
                """

                sfcrsToSend: "list[SFCRequest]" = []

                for i, sfcr in enumerate(self.sfcrs):
                    for c in range(exp["noOfCopies"]):
                        sfcrToSend: SFCRequest = sfcr.copy()
                        sfcrToSend["sfcrID"] = f"sfcr{i}-{c}"
                        sfcrsToSend.append(sfcrToSend)

                self._orchestrator.sendRequests(sfcrsToSend)

        trafficDesign: "list[TrafficDesign]" = [
            generateTrafficDesignFromFile(
                os.path.join(
                    f"{getConfig()['repoAbsolutePath']}",
                    "src",
                    "runs",
                    "hybrid",
                    "data",
                    "requests.csv",
                ),
                exp["trafficScale"],
                4,
                False,
                exp["trafficPattern"],
            )
        ]

        topology: Topology = generateFatTreeTopology(
            4, exp["linkBandwidth"], exp["noOfCPUs"], exp["memory"], 1
        )

        class HybridSolver(Solver):
            """
            Class to run the hybrid online-offline algorithm.
            """

            def generateEmbeddingGraphs(self):
                """
                Generate the embedding graphs.
                """

                try:
                    while self._requests.empty():
                        pass
                    requests: "list[EmbeddingGraph]" = []
                    while not self._requests.empty():
                        requests.append(self._requests.get())
                        sleep(0.1)

                    if mutation:
                        for mutPb in mutationProbabilities:
                            for indPb in individualProbabilities:
                                for i in range(noOfRuns):
                                    solve(
                                        requests,
                                        self._orchestrator.sendEmbeddingGraphs,
                                        self._orchestrator.deleteEmbeddingGraphs,
                                        trafficDesign,
                                        self._trafficGenerator,
                                        self._orchestrator.getTelemetry(),
                                        topology,
                                        "genesis",
                                        f"{exp['name']}_mutPb_{mutPb}_indPb_{indPb}_{i}",
                                        mutPb=mutPb,
                                        indPb=indPb,
                                        evaluateOnline=False
                                    )
                    elif cx:
                        for cxPb in crossoverProbabilities:
                            for i in range(noOfRuns):
                                solve(
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    topology,
                                    "genesis",
                                    f"{exp['name']}_cxPb_{cxPb}_{i}",
                                    cxPb=cxPb,
                                    evaluateOnline=False
                                )
                    elif sigma:
                        for sigmaVal in sigmas:
                                for i in range(noOfRuns):
                                    solve(
                                        requests,
                                        self._orchestrator.sendEmbeddingGraphs,
                                        self._orchestrator.deleteEmbeddingGraphs,
                                        trafficDesign,
                                        self._trafficGenerator,
                                        self._orchestrator.getTelemetry(),
                                        topology,
                                        "genesis",
                                        f"{exp['name']}_sigma_{sigmaVal}_{i}",
                                        sigma=sigmaVal,
                                        evaluateOnline=False
                                    )
                    elif rr:
                        for rejectionRate in rejectionRates:
                            for i in range(noOfRuns):
                                solve(
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    topology,
                                    "genesis",
                                    f"{exp['name']}_rejectionRate_{rejectionRate}_{i}",
                                    rejectionRate=rejectionRate,
                                    evaluateOnline=False
                                )
                    elif activation:
                        for activationFunction in activations:
                            for i in range(noOfRuns):
                                solve(
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    topology,
                                    "genesis",
                                    f"{exp['name']}_activation_{activationFunction}_{i}",
                                    activation=activationFunction,
                                    evaluateOnline=False
                                )
                    elif init:
                        for initLimitValue in initLimit:
                            for i in range(noOfRuns):
                                solve(
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    topology,
                                    "genesis",
                                    f"{exp['name']}_initLimit_{initLimitValue}_{i}",
                                    initLimit=initLimitValue,
                                    evaluateOnline=False
                                )
                    elif chain or dijkstra or gaussian:
                        for i in range(noOfRuns):
                            solve(
                                requests,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                trafficDesign,
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                topology,
                                "genesis",
                                f"{exp['name']}_chain_{chain}_dijkstra_{dijkstra}_gaussian_{gaussian}_{i}",
                                staticChain=chain,
                                dijkstra=dijkstra,
                                disableGaussian=not gaussian,
                                evaluateOnline=False
                            )
                    else:
                        for i in range(noOfRuns):
                            solve(
                                requests,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                trafficDesign,
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                topology,
                                "genesis",
                                f"{exp['name']}_{i}",
                                mutPb=MUT_PB,
                                indPb=GENE_MUT_PB,
                                cxPb=CX_PB,
                            )


                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        sfcEm: SFCEmulator = SFCEmulator(SFCRGen, HybridSolver, headless)
        sfcEm.startTest(
            topology,
            trafficDesign,
        )
        sfcEm.end()
