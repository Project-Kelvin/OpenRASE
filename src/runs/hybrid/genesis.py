"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import json
import os
from time import sleep
from typing import Any
import click
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


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--ga", is_flag=True, default=False, help="Run in GA hyperparameter tuning mode.")
@click.option("--genesis", is_flag=True, default=False, help="Run in GENESIS hyperparameter tuning mode.")
def run(headless: bool, ga: bool, genesis: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        ga (bool): Whether to run in GA hyperparameter tuning mode.
        genesis (bool): Whether to run in GENESIS hyperparameter tuning mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    rejectionRates: list[float] = [0.05, 0.07, 0.1]
    sigmas: list[float] = [0.0, 2.0, 5.0, 10.0]

    experimentsIncludeFilter: list[dict[str, Any]] = [
        (12, 0.2, False, 5, 1), # Hard
        (8, 0.2, False, 10, 2), # Medium
        (8, 0.1, False, 10, 2), # Easy
    ]

    if ga or genesis:
        experimentsIncludeFilter = [experimentsIncludeFilter[1]]  # Only run the medium experiment for hyperparameter tuning

    noOfRuns: int = 20

    experimentsExcludeFilter: list[dict[str, Any]] = []
    experimentPriority: list[str] = []
    experimentsToRun: list[dict[str, Any]] = []

    for noOfCopy in [12, 8]:
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

                    if ga:
                        for crossPb in crossoverProbabilities:
                            for mutPb in mutationProbabilities:
                                for indPb in individualProbabilities:
                                    for i in range(20):
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
                                            cxPb=crossPb,
                                            evaluateOnline=False
                                        )
                    elif genesis:
                        for sigma in sigmas:
                            for rejectionRate in rejectionRates:
                                for i in range(5):
                                    solve(
                                        requests,
                                        self._orchestrator.sendEmbeddingGraphs,
                                        self._orchestrator.deleteEmbeddingGraphs,
                                        trafficDesign,
                                        self._trafficGenerator,
                                        self._orchestrator.getTelemetry(),
                                        topology,
                                        "genesis",
                                        f"{exp['name']}_sigma_{sigma}_rejectionRate_{rejectionRate}_{i}",
                                        sigma=sigma,
                                        rejectionRate=rejectionRate,
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
