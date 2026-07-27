"""
The defines the script to run the hybrid online-offline algorithm.
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
from algorithms.hybrid.gaha import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology, generateTopologyFromEdgeList
from utils.traffic_design import generateTrafficDesignFromFile, generateTrafficDesignFromIoTTrace
from utils.tui import TUI


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--mutation", is_flag=True, default=False, help="Run in mutation probability hyperparameter tuning mode.")
@click.option("--cx", is_flag=True, default=False, help="Run in crossover probability hyperparameter tuning mode.")
@click.option("--env", type=click.Choice(["dc", "milan", "25n50e"]), default="dc", help="Network environment to run the algorithm in.")
@click.option("--offline", is_flag=True, default=False, help="Run in offline mode.")
def run(headless: bool, mutation: bool, cx: bool, env: str, offline: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        mutation (bool): Whether to run in mutation probability hyperparameter tuning mode.
        cx (bool): Whether to run in crossover probability hyperparameter tuning mode.
        env (str): The network environment to run the algorithm in.
        offline (bool): Whether to run in offline mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    delay: int = 1

    # 12, 0.3, False, 5, 1
    # 3, 0.2, False, 10, 0.5
    experimentsIncludeFilter: list[tuple[int, float, bool, int, int]] = [
        (20, 0.1, False, 10, 1), # Hard
        (12, 0.1, False, 10, 2), # Medium
        (8, 0.1, False, 10, 2), # Easy
    ]

    if mutation or cx:
        experimentsIncludeFilter = [experimentsIncludeFilter[0]]  # Only run the hard experiment for hyperparameter tuning

    noOfRuns: int = 20

    experimentsExcludeFilter: list[tuple[int, float, bool, int, float]] = [
    ]
    experimentPriority: list[str] = [
    ]
    experimentsToRun: list[dict[str, Any]] = []

    for noOfCopy in [20, 12, 8]:
        for trafficScale in [0.1, 0.2]:
            for trafficPattern in [False, True]:
                for linkBandwidth in [10, 5]:
                    for noOfCPUs in [2, 1, 0.5]:
                        experimentsToRun.append(
                            {
                                "name": f"{noOfCopy}_{trafficScale}_{trafficPattern}_{linkBandwidth}_{noOfCPUs}_{delay}",
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
                        "sfcrs.json",
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


        if env == "dc":
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
                4, exp["linkBandwidth"], exp["noOfCPUs"], exp["memory"], delay
            )
        else:
            trafficDesign: list[TrafficDesign] = [generateTrafficDesignFromIoTTrace(
                os.path.join(
                    f"{getConfig()['repoAbsolutePath']}",
                    "src",
                    "runs",
                    "hybrid",
                    "data",
                    "iot-trace.csv",
                ),
                60,
                10000,
            )]

            if env == "milan":
                topology: Topology = generateTopologyFromEdgeList(
                    os.path.join(
                        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "milan.txt"
                    ),
                    exp["noOfCPUs"],
                    exp["memory"],
                    exp["linkBandwidth"],
                    delay
                )
            elif env == "25n50e":
                topology: Topology = generateTopologyFromEdgeList(
                    os.path.join(
                        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "25n50e.txt"
                    ),
                    exp["noOfCPUs"],
                    exp["memory"],
                    exp["linkBandwidth"],
                    delay
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
                                        f"{exp['name']}_mutPb{mutPb}_indPb{indPb}_{i}",
                                        mutPb = mutPb,
                                        indPb = indPb,
                                        evaluateOnline = False,
                                    )
                    elif cx:
                        for cxPb in crossoverProbabilities:
                            for i in range(noOfRuns):
                                TUI.appendToSolverLog(
                                    f"Running experiment {exp['name']} with cxPb={cxPb}."
                                )
                                solve(
                                    topology,
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    f"{exp['name']}_cxPb{cxPb}_{i}",
                                    cxPb = cxPb,
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
                                evaluateOnline = not offline
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
