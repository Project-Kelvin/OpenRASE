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
from algorithms.hybrid.genesis import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
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
@click.option("--rr", is_flag=True, default=False, help="Run in Rejection Rate tuning mode.")
@click.option("--sigma", is_flag=True, default=False, help="Run in sigma hyperparameter tuning mode.")
@click.option("--chain", is_flag=True, default=False, help="Use static chain decoding.")
@click.option("--dijkstra", is_flag=True, default=False, help="Use Dijkstra's algorithm for pathfinding.")
@click.option("--gaussian", is_flag=True, default=False, help="Disable the Gaussian distribution for host selection.")
@click.option("--activation", is_flag=True, default=False, help="Test activation functions in the neural network.")
@click.option("--init", is_flag=True, default=False, help="Test the limit to use for generating the predefined weights.")
@click.option("--env", type=click.Choice(["dc", "milan", "25n50e"], case_sensitive=False), default="dc", help="The network environment to run the algorithm in.")
@click.option("--retrain", is_flag=True, default=False, help="Specifies if BENNS should be retrained.")
@click.option("--offline", is_flag=True, default=False, help="Run in offline mode.")
@click.option("--test", is_flag=True, default=False, help="Run in test mode.")
@click.option("--random-input-weights", is_flag=True, default=False, help="Use random input weights instead of predefined weights.")
@click.option("--neurons", is_flag=True, default=False, help="Test the number of neurons in the neural network.")
@click.option("--random-host", is_flag=True, default=False, help="Use random host ids instead of the ones in the topology.")
@click.option("--himode", type=click.Choice(["hard", "easy"], case_sensitive=False), default="hard", help="Run in hi or genesis mode.")
def run(headless: bool, mutation: bool, cx: bool, rr: bool, sigma: bool, chain: bool, dijkstra: bool, gaussian: bool, activation: str, init: bool, env: str, retrain: bool, offline: bool, test: bool, random_input_weights: bool, neurons: bool, random_host: bool, himode: str) -> None:
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
        env (str): The network environment to run the algorithm in.
        retrain (bool): Whether to retrain the BENNS model.
        offline (bool): Whether to run in offline mode.
        test (bool): Whether to run in test mode.
        random_input_weights (bool): Whether to use random input weights instead of predefined weights.
        neurons (bool): Whether to test the number of neurons in the neural network.
        random_host (bool): Whether to use random host ids instead of the ones in the topology.
        himode (str): Whether to run in hi or genesis mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    rejectionRates: list[float] = [0.0, 0.05, 0.07, 0.1]
    sigmas: list[float] = [0.0, 1.0, 2.0, 4.0]
    activations: list[str] = ["tanh", "sin", "relu", "linear"]
    initLimit: list[float] = [1, 2, np.pi, 2 * np.pi]
    noOfNeurons: list[int] = [1, 4, 6]
    delay: int = 1

    experiments: list[tuple[int, float, bool, float, float]] = [
        (15, 0.23, False, 5, 0.5), # Used for ablation (DC)
        (20, 0.1, False, 10, 1), # Used for hyperparameter tuning (DC),
        (20, 0.1, False, 10, 1), # Used VNF embedding only experiment (Milan).
        (10, 0.1, False, 10, 1), # Used VNF embedding only experiment (25N50E)
        (8, 0.1, False, 10, 2), # Used for hyperparameter tuning in BEGA,
        (8, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Easy,
        (12, 0.3, False, 5, 0.25), # Used for hyperparameter tuning in HiGENESIS Hard,
    ]

    if mutation or cx or rr or sigma or activation:
        selectedExperiments = [experiments[1]]

    if init or chain or dijkstra or gaussian or neurons or random_input_weights or random_host or env == "dc":
        selectedExperiments = [experiments[0]]

    if env == "milan":
        if (rr or sigma) and himode == "easy":
            selectedExperiments = [experiments[5]] # HiGENESIS sigma and rr tuning
        elif (rr or sigma) and himode == "hard":
            selectedExperiments = [experiments[6]] # HiGENESIS sigma and rr tuning
        else:
            selectedExperiments = [experiments[2]]

    if env == "25n50e":
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
                4, exp["linkBandwidth"], exp["noOfCPUs"], exp["memory"], delay, randomHost=random_host
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
                            "genesis",
                            f"{exp['name']}_{i}",
                            retrain=retrain,
                            evaluateOnline = not offline,
                            linesToWrite=linesToWrite
                        )
                except Exception as e:
                    TUI.appendToSolverLog(str(e), True)

                TUI.appendToSolverLog("Finished experiment.")

        if mutation or cx or rr or sigma or activation or init or chain or dijkstra or gaussian or test or random_input_weights or neurons:
            TUI.disable()
            sfcrsToSend: "list[SFCRequest]" = generateSFCRs(exp["noOfCopies"])

            if mutation:
                for mutPb in mutationProbabilities:
                    for indPb in individualProbabilities:
                        for i in range(noOfRuns):
                            seed: int = setRandomSeed()
                            linesToWrite: list[str] = [
                                f"Seed: {seed}",
                            ]
                            solve(
                                sfcrsToSend,
                                None,
                                None,
                                trafficDesign,
                                None,
                                None,
                                topology,
                                "genesis",
                                f"{exp['name']}_mutPb_{mutPb}_indPb_{indPb}_{i}",
                                mutPb=mutPb,
                                indPb=indPb,
                                evaluateOnline=False,
                                linesToWrite=linesToWrite
                            )
            elif cx:
                for cxPb in crossoverProbabilities:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
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
                            "genesis",
                            f"{exp['name']}_cxPb_{cxPb}_{i}",
                            cxPb=cxPb,
                            evaluateOnline=False,
                            linesToWrite=linesToWrite
                        )
            elif sigma:
                for sigmaVal in sigmas:
                        for i in range(noOfRuns):
                            seed: int = setRandomSeed()
                            linesToWrite: list[str] = [
                                f"Seed: {seed}",
                            ]
                            solve(
                                sfcrsToSend,
                                None,
                                None,
                                trafficDesign,
                                None,
                                None,
                                topology,
                                "genesis",
                                f"{exp['name']}_sigma_{sigmaVal}_{i}",
                                sigma=sigmaVal,
                                evaluateOnline=False,
                                linesToWrite=linesToWrite
                            )
            elif rr:
                for rejectionRate in rejectionRates:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
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
                            "genesis",
                            f"{exp['name']}_rejectionRate_{rejectionRate}_{i}",
                            rejectionRate=rejectionRate,
                            evaluateOnline=False,
                            linesToWrite=linesToWrite
                        )
            elif activation:
                for activationFunction in activations:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
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
                            "genesis",
                            f"{exp['name']}_activation_{activationFunction}_{i}",
                            activation=activationFunction,
                            evaluateOnline=False,
                            linesToWrite=linesToWrite
                        )
            elif init:
                for initLimitValue in initLimit:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
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
                            "genesis",
                            f"{exp['name']}_initLimit_{initLimitValue}_{i}",
                            initLimit=initLimitValue,
                            evaluateOnline=False,
                            linesToWrite=linesToWrite
                        )
            elif chain or dijkstra or gaussian or random_input_weights:
                for i in range(noOfRuns):
                    seed: int = setRandomSeed()
                    linesToWrite: list[str] = [
                        f"Seed: {seed}",
                    ]
                    solve(
                        sfcrsToSend,
                        None,
                        None,
                        trafficDesign,
                        None,
                        None,
                        topology,
                        "genesis",
                        f"{exp['name']}_chain_{chain}_dijkstra_{dijkstra}_gaussian_{gaussian}_random_input_weights_{random_input_weights}_{i}",
                        staticChain=chain,
                        dijkstra=dijkstra,
                        disableGaussian=gaussian,
                        randomInputWeights=random_input_weights,
                        evaluateOnline=False,
                        linesToWrite=linesToWrite
                    )
            elif test:
                TUI.appendToSolverLog(f"Running experiment {exp['name']} in test mode.")
                for i in range(noOfRuns):
                    seed: int = setRandomSeed()
                    linesToWrite: list[str] = [
                        f"Seed: {seed}",
                    ]
                    solve(
                        sfcrsToSend,
                        None,
                        None,
                        trafficDesign,
                        None,
                        None,
                        topology,
                        "genesis",
                        f"{exp['name']}_{i}",
                        retrain=False,
                        evaluateOnline = False,
                        linesToWrite=linesToWrite
                    )
            elif neurons:
                for noOfNeuronsValue in noOfNeurons:
                    for i in range(noOfRuns):
                        seed: int = setRandomSeed()
                        linesToWrite: list[str] = [
                            f"Seed: {seed}",
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
                            "genesis_neurons",
                            f"{exp['name']}_noOfNeurons_{noOfNeuronsValue}_{i}",
                            noOfNeurons=noOfNeuronsValue,
                            evaluateOnline=False,
                            linesToWrite=linesToWrite
                        )
        else:
            sfcEm: SFCEmulator = SFCEmulator(SFCRGen, HybridSolver, headless)
            sfcEm.startTest(
                topology,
                trafficDesign,
            )
            sfcEm.end()
