"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import json
import os
import random
from time import sleep
from typing import Any
import click
import numpy as np
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.bega import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology, generateTopologyFromEdgeList
from utils.traffic_design import generateTrafficDesignFromFile, generateTrafficDesignFromIoTTrace
from utils.tui import TUI

def setRandomSeed() -> int:
    """
    Sets a random seed for the experiment.

    Returns:
        int: the random seed.
    """

    seed: int = random.randint(0, 10000000)

    random.seed(seed)
    np.random.seed(seed)

    return seed

@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--mutation", is_flag=True, default=False, help="Run in mutation probability hyperparameter tuning mode.")
@click.option("--cx", is_flag=True, default=False, help="Run in crossover probability hyperparameter tuning mode.")
@click.option("--env", type=click.Choice(["dc", "milan", "25n50e"], case_sensitive=False), default="dc", help="Environment to run the experiments in.")
def run(headless: bool, mutation: bool, cx: bool, env: str) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        mutation (bool): Whether to run in mutation probability hyperparameter tuning mode.
        cx (bool): Whether to run in crossover probability hyperparameter tuning mode.
        env (str): The environment to run the experiments in.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    delay: int = 1

    #(15, 0.1, False, 10, 0.1) works
    experiments: list[tuple[int, float, bool, float, float]] = [
        (15, 0.23, False, 5, 0.5), # Used for ablation (DC)
        (20, 0.1, False, 10, 1), # Used for hyperparameter tuning (DC),
        (20, 0.1, False, 10, 1), # Used VNF embedding only experiment (Milan)
        (10, 0.1, False, 10, 1), # Used VNF embedding only experiment (25N50E)
    ]

    if mutation or cx:
        experiments = [experiments[1]]

    if env == "dc":
        experiments = [experiments[0]]

    if env == "milan":
        experiments = [experiments[2]]

    if env == "25n50e":
        experiments = [experiments[3]]

    noOfRuns: int = 20

    for experiment in experiments:
        noOfCopy, trafficScale, trafficPattern, linkBandwidth, noOfCPUs = experiment
        exp: dict[str, Any] = dict(
            {
                "name": f"{noOfCopy}_{trafficScale}_{trafficPattern}_{linkBandwidth}_{noOfCPUs}_{delay}_{env}",
                "noOfCopies": noOfCopy,
                "trafficScale": trafficScale,
                "trafficPattern": trafficPattern,
                "linkBandwidth": linkBandwidth,
                "noOfCPUs": noOfCPUs,
                "memory": 5120,
            }
        )
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
                30,
                1000 / exp["trafficScale"],
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
            if env == "25n50e":
                topology: Topology = generateTopologyFromEdgeList(
                    os.path.join(
                        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "25N50E.txt"
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
                                    seed: int = setRandomSeed()
                                    linesToWrite: list[str] = [
                                        f"Seed: {seed}",
                                    ]
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
                                        linesToWrite=linesToWrite
                                    )
                    elif cx:
                        for cxPb in crossoverProbabilities:
                            for i in range(noOfRuns):
                                TUI.appendToSolverLog(
                                    f"Running experiment {exp['name']} with cxPb={cxPb}."
                                )
                                seed: int = setRandomSeed()
                                linesToWrite: list[str] = [
                                    f"Seed: {seed}",
                                ]
                                solve(
                                    topology,
                                    requests,
                                    self._orchestrator.sendEmbeddingGraphs,
                                    self._orchestrator.deleteEmbeddingGraphs,
                                    trafficDesign,
                                    self._trafficGenerator,
                                    self._orchestrator.getTelemetry(),
                                    f"{exp['name']}_cxPb{cxPb}_{i}",
                                    cxpPb = cxPb,
                                    evaluateOnline = False,
                                    linesToWrite=linesToWrite
                                )
                    else:
                        TUI.appendToSolverLog(
                            f"Running experiment {exp['name']} with default parameters."
                        )

                        for i in range(noOfRuns):
                            TUI.appendToSolverLog(
                                f"Running experiment round {i}."
                            )
                            seed: int = setRandomSeed()
                            linesToWrite: list[str] = [
                                f"Seed: {seed}",
                            ]
                            solve(
                                topology,
                                requests,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                trafficDesign,
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                f"{exp['name']}_{i}",
                                linesToWrite=linesToWrite
                            )

                except Exception as e:
                    TUI.appendToSolverLog(f"Error: {str(e)}", True)

                TUI.appendToSolverLog("Finished experiment.")

        sfcEm: SFCEmulator = SFCEmulator(FGGen, HybridSolver, headless)
        sfcEm.startTest(
            topology,
            trafficDesign,
        )
        sfcEm.end()
