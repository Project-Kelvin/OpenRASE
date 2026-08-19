"""
The defines the script to run the hybrid online-offline algorithm.
"""

from copy import deepcopy
import json
import os
import random
from time import sleep
from typing import Any
import click
import numpy as np
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
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
@click.option("--env", type=click.Choice(["dc", "milan", "25n50e"]), default="dc", help="Network environment to run the algorithm in.")
@click.option("--offline", is_flag=True, default=False, help="Run in offline mode.")
@click.option("--retrain", is_flag=True, default=False, help="Retrain the surrogate model.")
@click.option("--test", is_flag=True, default=False, help="Run in test mode.")
def run(headless: bool, mutation: bool, cx: bool, env: str, offline: bool, retrain: bool, test: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        mutation (bool): Whether to run in mutation probability hyperparameter tuning mode.
        cx (bool): Whether to run in crossover probability hyperparameter tuning mode.
        env (str): The network environment to run the algorithm in.
        offline (bool): Whether to run in offline mode.
        retrain (bool): Whether to retrain the surrogate model.
        test (bool): Whether to run in test mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    delay: int = 1
    selectedExperiments: list[tuple[int, float, bool, float, float]] = []

    #(15, 0.1, False, 10, 0.1) works
    experiments: list[tuple[int, float, bool, float, float]] = [
        (15, 0.23, False, 5, 0.5), # Used for ablation (DC)
        (20, 0.1, False, 10, 1), # Used for hyperparameter tuning (DC),
        (20, 0.1, False, 10, 1), # Used VNF embedding only experiment (Milan)
        (10, 0.1, False, 10, 1), # Used VNF embedding only experiment (25N50E)
        (8, 0.1, False, 10, 2), # Used for hyperparameter tuning in BEGA
    ]

    if mutation or cx:
        selectedExperiments = [experiments[1]]

    elif env == "dc":
        selectedExperiments = [experiments[4]]

    elif env == "milan":
        selectedExperiments = [experiments[2]]

    elif env == "25n50e":
        selectedExperiments = [experiments[3]]

    noOfRuns: int = 20

    for experiment in selectedExperiments:
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
                        fgToSend: EmbeddingGraph = deepcopy(fg)
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
            elif env == "25n50e":
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

                    TUI.appendToSolverLog(
                        f"Running experiment {exp['name']} with default parameters."
                    )

                    for i in range(noOfRuns):
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
                            evaluateOnline = not offline,
                            retrain = retrain,
                            linesToWrite=linesToWrite
                        )

                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        if test or mutation or cx:
            TUI.disable()
            fgsToSend: "list[SFCRequest]" = []
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
                fgs = json.load(f)

                for i, fg in enumerate(fgs):
                    for c in range(exp["noOfCopies"]):
                        fgToSend: SFCRequest = deepcopy(fg)
                        fgToSend["sfcrID"] = f"sfcr{i}-{c}"
                        fgsToSend.append(fgToSend)

            TUI.appendToSolverLog(f"Running experiment {exp['name']} in test mode.")

            if mutation:
                for mutPb in mutationProbabilities:
                    for indPb in individualProbabilities:
                        for i in range(noOfRuns):
                            seed: int = setRandomSeed()
                            linesToWrite: list[str] = [
                                f"Seed: {seed}",
                            ]
                            TUI.appendToSolverLog(
                                f"Running experiment {exp['name']} with mutPb={mutPb} and indPb={indPb}."
                            )
                            solve(
                                topology,
                                fgsToSend,
                                None,
                                None,
                                trafficDesign,
                                None,
                                None,
                                f"{exp['name']}_mutPb{mutPb}_indPb{indPb}_{i}",
                                mutPb = mutPb,
                                indPb = indPb,
                                evaluateOnline = False,
                                linesToWrite=linesToWrite,
                                dirName="rega_mutation"
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
                            fgsToSend,
                            None,
                            None,
                            trafficDesign,
                            None,
                            None,
                            f"{exp['name']}_cxPb{cxPb}_{i}",
                            cxPb = cxPb,
                            evaluateOnline = False,
                            linesToWrite=linesToWrite,
                            dirName="rega_cx"
                        )
            if test:
                for i in range(noOfRuns):
                    seed: int = setRandomSeed()
                    linesToWrite: list[str] = [
                        f"Seed: {seed}",
                    ]
                    solve(
                        topology,
                        fgsToSend,
                        None,
                        None,
                        trafficDesign,
                        None,
                        None,
                        f"{exp['name']}_{i}",
                        evaluateOnline = False,
                        retrain = False,
                        linesToWrite=linesToWrite
                    )
        else:
            sfcEm: SFCEmulator = SFCEmulator(FGGen, HybridSolver, headless)
            sfcEm.startTest(
                topology,
                trafficDesign,
            )
            sfcEm.end()
