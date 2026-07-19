"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import json
import os
from time import sleep
from typing import Any
import click
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.bega import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--hyper", is_flag=True, default=False, help="Run in hyperparameter tuning mode.")
def run(headless: bool, hyper: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        hyper (bool): Whether to run in hyperparameter tuning mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]

    experimentsIncludeFilter: list[tuple[int, float, bool, int, int]] = [
        (25, 0.1, False, 10, 1), # Hard
        (12, 0.1, False, 10, 2), # Medium
        (8, 0.1, False, 10, 2), # Easy
    ]

    if hyper:
        experimentsIncludeFilter = [experimentsIncludeFilter[2]]  # Only run the easy experiment for hyperparameter tuning

    noOfRuns: int = 20

    experimentsExcludeFilter: list[tuple[int, float, bool, int, float]] = [
        # (16, 0.1, False, 5, 1),
        # (16, 0.1, False, 5, 2),
        # (16, 0.1, False, 10, 1),
        # (16, 0.1, False, 10, 2),
        # (16, 0.1, True, 5, 1),
        # (16, 0.1, True, 5, 2),
        # (16, 0.1, True, 10, 1),
        # (16, 0.1, True, 10, 2),
        # (16, 0.2, False, 5, 1), #incomplete
        # (16, 0.2, False, 5, 2), #incomplete
        # (16, 0.2, True, 5, 1), #not done
        # (16, 0.2, True, 5, 2), #not done
        # (16, 0.2, False, 10, 1),
        # (16, 0.2, False, 10, 2),
        # (16, 0.2, True, 10, 1), #incomplete
        # (16, 0.2, True, 10, 2), #incomplete,
    ]
    experimentPriority: list[str] = [
    ]
    experimentsToRun: list[dict[str, Any]] = []

    for noOfCopy in [25, 12, 8]:
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
            key=lambda x: experimentPriority.index(x["name"])
            if x["name"] in experimentPriority
            else len(experimentPriority),
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

        class FGGen(FGRequestGenerator):
            """
            Class to generate FG Requests.
            """

            def __init__(self, orchestrator: Orchestrator) -> None:
                """
                Initialize the FGGen class.
                """

                super().__init__(orchestrator)
                with open(
                    os.path.join(
                        getConfig()["repoAbsolutePath"],
                        "src",
                        "runs",
                        "hybrid",
                        "configs",
                        "forwarding-graphs.json",
                    ),
                    "r",
                    encoding="utf8",
                ) as f:
                    self.fgs = json.load(f)

            def generateRequests(self) -> "list[EmbeddingGraph]":
                """
                Generate the FG Requests.
                """

                fgsToSend: "list[EmbeddingGraph]" = []

                for i, fg in enumerate(self.fgs):
                    for c in range(exp["noOfCopies"]):
                        fgToSend: EmbeddingGraph = fg.copy()
                        fgToSend["sfcrID"] = f"sfcr{i}-{c}"
                        fgsToSend.append(fgToSend)

                self._orchestrator.sendRequests(fgsToSend)

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

                    if hyper:
                        for crossPb in crossoverProbabilities:
                            for mutPb in mutationProbabilities:
                                for indPb in individualProbabilities:
                                    for i in range(20):
                                        TUI.appendToSolverLog(
                                            f"Running experiment {exp['name']} with mutPb={mutPb} and indPb={indPb}."
                                        )
                                        solve(
                                            topology,
                                            requests,
                                            self._orchestrator.sendEmbeddingGraphs,
                                            self._orchestrator.deleteEmbeddingGraphs,
                                            trafficDesign,
                                            self._trafficGenerator,
                                            self._orchestrator.getTelemetry(),
                                            f"{exp['name']}_mutPb{mutPb}_indPb{indPb}_cxPb{crossPb}_{i}",
                                            mutPb = mutPb,
                                            indPb = indPb,
                                            cxpPb = crossPb,
                                            evaluateOnline = False,
                                        )
                    else:
                        TUI.appendToSolverLog(
                            f"Running experiment {exp['name']} with default parameters."
                        )

                        for i in range(noOfRuns):
                            solve(
                                topology,
                                requests,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                trafficDesign,
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                f"{exp['name']}_{i}",
                            )

                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        sfcEm: SFCEmulator = SFCEmulator(FGGen, HybridSolver, headless)
        sfcEm.startTest(
            topology,
            trafficDesign,
        )
        sfcEm.end()
