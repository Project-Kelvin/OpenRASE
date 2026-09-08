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
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.hi_genesis import solve
from mano.orchestrator import Orchestrator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology, generateTopologyFromEdgeList
from utils.traffic_design import generateTrafficDesignFromFile, generateTrafficDesignFromIoTTrace
from utils.tui import TUI

# Tuned Hyperparameters for HiGENESIS
# Individual Mutation Probability: 0.7
# Gene Mutation Probability: 0.7
# Crossover Probability: 1.0
# Dominance Threshold: 0.5

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

def generateSFCRs(noOfCopies: int) -> "list[SFCRequest]":
    """
    Generate the SFC Requests.

    Parameters:
        noOfCopies (int): The number of copies of each SFC Request to generate.

    Returns:
        list[SFCRequest]: A list of SFC Requests.
    """

    sfcrsToSend: "list[SFCRequest]" = []
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
        sfcrs = json.load(f)

        for i, sfcr in enumerate(sfcrs):
            for c in range(noOfCopies):
                sfcrToSend: SFCRequest = sfcr.copy()
                sfcrToSend["sfcrID"] = f"sfcr{i}-{c}"
                sfcrsToSend.append(sfcrToSend)

    return sfcrsToSend


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--mutation", is_flag=True, default=False, help="Run in mutation pbs hyperparameter tuning mode.")
@click.option("--cx", is_flag=True, default=False, help="Run in crossover pb hyperparameter tuning mode.")
@click.option("--root", is_flag=True, default=False, help="Run in root hyperparameter tuning mode.")
@click.option("--test", is_flag=True, default=False, help="Run in test mode.")
@click.option("--himode", type=click.Choice(["hard", "easy", "medium", "harder", "hardest", "beast"], case_sensitive=False), default="hard", help="Run in hi or genesis mode.")
@click.option("--runs", type=int, default=20, help="Number of test runs.")
@click.option("--static", is_flag=True, default=False, help="Run in static root mode.")
@click.option("--root-value", type=int, default=4, help="Root individual value for static root mode.")
@click.option("--mutation-value", type=float, default=0.7, help="Individual mutation probability.")
@click.option("--gene-value", type=float, default=0.7, help="Gene mutation probability.")
@click.option("--crossover-value", type=float, default=1.0, help="Crossover probability.")
@click.option("--tune", is_flag=True, default=False, help="Run in hyperparameter tuning mode.")
@click.option("--dom-thresh", type=float, default=0.5, help="Dominance threshold for hyperparameter tuning.")
def run(headless: bool, mutation: bool, cx: bool, root: bool, test: bool, himode: str, runs: int, static: bool, root_value: int, mutation_value: float, gene_value: float, crossover_value: float, tune: bool, dom_thresh: float) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        mutation (bool): Whether to run in mutation pbs hyperparameter tuning mode.
        cx (bool): Whether to run in crossover pb hyperparameter tuning mode.
        root (bool): Whether to run in root hyperparameter tuning mode.
        test (bool): Whether to run in test mode.
        himode (str): Whether to run in hi or genesis mode.
        runs (int): Number of test runs.
        static (bool): Whether to run in static root mode.
        root_value (int): The value of the root individual for static root mode.
        mutation_value (float): The individual mutation probability.
        gene_value (float): The gene mutation probability.
        crossover_value (float): The crossover probability.
        tune (bool): Whether to run in hyperparameter tuning mode.
        dom_thresh (float): The dominance threshold for hyperparameter tuning.

    Returns:
        None
    """

    delay: int = 1
    selectedExperiments: list[tuple[int, float, bool, float, float]] = []
    dirName: str = "higenesis"
    minAR: float = 0.95
    env: str = "milan"

    metaCxPb: float = crossover_value
    metaIndPb: float = mutation_value
    metaGenePb: float = gene_value

    geneMutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualMutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    roots: list[int] = [1, 2, 4, 5, 10, 20]
    rootIndividual: int = root_value if static else -1

    experiments: list[tuple[int, float, bool, float, float]] = [
        (15, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Easy,
        (17, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Medium,
        (19, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Hard,
        (21, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Harder,
        (23, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Hardest,
        (25, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Beast,
    ]

    if himode == "easy":
        selectedExperiments = [experiments[0]]
        dirName = "higenesis_easy"
        minAR = 0.95
    elif himode == "medium":
        selectedExperiments = [experiments[1]]
        dirName = "higenesis_medium"
        minAR = 0.95
    elif himode == "hard":
        selectedExperiments = [experiments[2]]
        dirName = "higenesis_hard"
        minAR = 0.90
    elif himode == "harder":
        selectedExperiments = [experiments[3]]
        dirName = "higenesis_harder"
        minAR = 0.90
    elif himode == "hardest":
        selectedExperiments = [experiments[4]]
        dirName = "higenesis_hardest"
        minAR = 0.85
    elif himode == "beast":
        selectedExperiments = [experiments[5]]
        dirName = "higenesis_beast"
        minAR = 0.80
    else:
        selectedExperiments = [experiments[2]]

    if static:
        dirName = f"{dirName}_static_{root_value}"

    noOfRuns: int = runs

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

        class SFCRGen(SFCRequestGenerator):
            """
            Class to generate FG Requests.
            """

            def __init__(self, orchestrator: Orchestrator) -> None:
                """
                Initialize the SFCRGen class.
                """

                super().__init__(orchestrator)

            def generateRequests(self) -> None:
                """
                Generate the FG Requests.
                """

                self._orchestrator.sendRequests(generateSFCRs(exp["noOfCopies"]))


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

                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
                            f"Environment: {env}",
                        ]
                        solve(
                            requests,
                            self._orchestrator.sendEmbeddingGraphs,
                            self._orchestrator.deleteEmbeddingGraphs,
                            trafficDesign,
                            self._trafficGenerator,
                            self._orchestrator.getTelemetry(),
                            topology,
                            dirName,
                            f"{exp['name']}_{i}",
                            linesToWrite=linesToWrite,
                            minimumAR=minAR,
                            cxPb=metaCxPb,
                            indPb=metaGenePb,
                            mutPb=metaIndPb,
                            dominanceThreshold=dom_thresh,
                        )
                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        if test or mutation or cx or root:
            TUI.disable()
            sfcrsToSend: "list[SFCRequest]" = generateSFCRs(exp["noOfCopies"])

            if test:
                TUI.appendToSolverLog(f"Running experiment {exp['name']} in test mode.")
                for i in range(noOfRuns):
                    seed: int = setRandomSeed()
                    linesToWrite: list[str] = [
                        f"Seed: {seed}",
                        f"Environment: {env}",
                        f"Individual Mutation Probability: {metaIndPb}",
                        f"Gene Mutation Probability: {metaGenePb}",
                        f"Crossover Probability: {metaCxPb}",
                        f"Root Individual: {rootIndividual}"
                    ]
                    solve(
                        sfcrsToSend,
                        None,
                        None,
                        trafficDesign,
                        None,
                        None,
                        topology,
                        dirName,
                        f"{exp['name']}_{i}" if not tune else f"{exp['name']}_indPb{metaIndPb}_genePb{metaGenePb}_cxPb{metaCxPb}_domThresh{dom_thresh}_{i}",
                        evaluateOnline = False,
                        linesToWrite=linesToWrite,
                        minimumAR=minAR,
                        rootIndividual=rootIndividual,
                        cxPb=metaCxPb,
                        indPb=metaGenePb,
                        mutPb=metaIndPb,
                        dominanceThreshold=dom_thresh
                    )
            elif mutation:
                TUI.appendToSolverLog(f"Running experiment {exp['name']} in mutation hyperparameter tuning mode.")
                for indPb in individualMutationProbabilities:
                    for genePb in geneMutationProbabilities:
                        for i in range(noOfRuns):
                            seed: int = setRandomSeed()
                            linesToWrite: list[str] = [
                                f"Seed: {seed}",
                                f"Environment: {env}",
                                f"Individual Mutation Probability: {indPb}",
                                f"Gene Mutation Probability: {genePb}",
                            ]
                            solve(
                                sfcrsToSend,
                                None,
                                None,
                                trafficDesign,
                                None,
                                None,
                                topology,
                                dirName,
                                f"{exp['name']}_indpb_{indPb}_genepb_{genePb}_{i}",
                                mutPb=indPb,
                                indPb=genePb,
                                evaluateOnline = False,
                                linesToWrite=linesToWrite,
                                minimumAR=minAR,
                                rootIndividual=rootIndividual
                            )
            elif cx:
                TUI.appendToSolverLog(f"Running experiment {exp['name']} in crossover hyperparameter tuning mode.")
                for cxPb in crossoverProbabilities:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
                            f"Environment: {env}",
                            f"Crossover Probability: {cxPb}",
                        ]
                        solve(
                            sfcrsToSend,
                            None,
                            None,
                            trafficDesign,
                            None,
                            None,
                            topology,
                            dirName,
                            f"{exp['name']}_cxpb_{cxPb}_{i}",
                            cxPb=cxPb,
                            evaluateOnline = False,
                            linesToWrite=linesToWrite,
                            minimumAR=minAR,
                            rootIndividual=rootIndividual
                        )
            elif root:
                TUI.appendToSolverLog(f"Running experiment {exp['name']} in root hyperparameter tuning mode.")
                for rootInd in roots:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
                            f"Root: {rootInd}",
                            f"Environment: {env}",
                        ]
                        solve(
                            sfcrsToSend,
                            None,
                            None,
                            trafficDesign,
                            None,
                            None,
                            topology,
                            dirName,
                            f"{exp['name']}_root_{rootInd}_{i}",
                            evaluateOnline = False,
                            linesToWrite=linesToWrite,
                            minimumAR=minAR,
                            rootIndividual=rootInd,
                            cxPb=metaCxPb,
                            indPb=metaGenePb,
                            mutPb=metaIndPb
                        )
        else:
            sfcEm: SFCEmulator = SFCEmulator(SFCRGen, HybridSolver, headless)
            sfcEm.startTest(
                topology,
                trafficDesign,
            )
            sfcEm.end()
