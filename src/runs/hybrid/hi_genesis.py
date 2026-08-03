"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import copy
import json
import os
import random
from time import sleep
import click
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.hi_genesis import solve
from mano.orchestrator import Orchestrator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--client", is_flag=True, default=False, help="Run in client mode.")
@click.option("--hyper", is_flag=True, default=False, help="Run in hyperparameter tuning mode.")
def run(headless: bool, client: bool, hyper: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        client (bool): Whether to run the algorithm in client mode.
        hyper (bool): Whether to run the algorithm in hyperparameter tuning mode.

    Returns:
        None
    """

    mutationProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    individualProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    crossoverProbabilities: list[float] = [0.2, 0.5, 0.7, 1.0]
    noOfRuns: int = 20

    design: TrafficDesign = generateTrafficDesignFromFile(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "hybrid",
            "data",
            "requests.csv",
        ),
        1,
        20,
        False,
    )

    steps: int = len(design)
    segments: int = 10 if not hyper else 1
    stepsPerSegment: int = steps // segments
    trafficSegments: "list[TrafficDesign]" = []
    for segment in range(segments):
        startStep: int = segment * stepsPerSegment
        endStep: int = (segment + 1) * stepsPerSegment

        segmentDesign: TrafficDesign = design[startStep:endStep]
        trafficSegments.append(segmentDesign)

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

        def generateRequests(self) -> None:
            """
            Generate the FG Requests.
            """

            sfcrsToSend: "list[SFCRequest]" = []

            for i, sfcr in enumerate(self.sfcrs):
                sfcrToSend: SFCRequest = sfcr.copy()
                sfcrToSend["sfcrID"] = f"sfcr{i}"
                sfcrsToSend.append(sfcrToSend)

            self._orchestrator.sendRequests(sfcrsToSend)

    topology: Topology = generateFatTreeTopology(
        4, 10, 1, 5120, 1
    )

    def removeHost(topology: Topology, hostID: str) -> Topology:
        """
        Remove a host from the topology.

        Parameters:
            topology (Topology): The topology.
            hostID (str): The ID of the host to remove.

        Returns:
            Topology: The topology without the host.
        """

        newTopology: Topology = copy.deepcopy(topology)

        newTopology["hosts"] = [host for host in newTopology["hosts"] if host["id"] != hostID]
        newTopology["links"] = [
            link for link in newTopology["links"]
            if link["source"] != hostID and link["destination"] != hostID
        ]

        return newTopology

    class HybridSolver(Solver):
        """
        Class to run the hybrid online-offline algorithm.
        """

        def generateEmbeddingGraphs(self):
            """
            Generate the embedding graphs.
            """

            allRequestsReceived: "list[SFCRequest]" = []
            originalRequests: "list[SFCRequest]" = []
            removedHosts: "list[int]" = []
            try:
                topologyToUse: Topology = copy.deepcopy(topology)
                while self._requests.empty():
                    pass

                while not self._requests.empty():
                    originalRequests.append(self._requests.get())
                    sleep(0.1)

                for segment in range(segments):
                    for request in originalRequests:
                        copies: int = 20 if segment == 0 else 1
                        for c in range(copies):
                            requestCopy: SFCRequest = copy.deepcopy(request)
                            requestCopy["sfcrID"] = f"{request['sfcrID']}-{c}-{segment}"
                            allRequestsReceived.append(requestCopy)

                    if segment > 5:
                        # Simulate a host failure
                        hosts: list[int] = [i for i in range(len(topologyToUse["hosts"])) if i not in removedHosts]
                        hostIdToRemove: int = random.choice(hosts)
                        hostToRemove: str = f"host{hostIdToRemove}"
                        removedHosts.append(hostIdToRemove)
                        topologyToUse = removeHost(topologyToUse, hostToRemove)
                        TUI.appendToSolverLog(f"Simulated failure of host {hostToRemove}.")

                    # if segment < 9:
                    #     continue

                    self._trafficGenerator.setDesign([trafficSegments[segment]])

                    if hyper:
                        for crossPb in crossoverProbabilities:
                            for mutPb in mutationProbabilities:
                                for indPb in individualProbabilities:
                                    for i in range(noOfRuns):
                                        TUI.appendToSolverLog(
                                            f"Running experiment {segment} with mutPb={mutPb} and indPb={indPb}."
                                        )
                                        solve(
                                            allRequestsReceived,
                                            self._orchestrator.sendEmbeddingGraphs,
                                            self._orchestrator.deleteEmbeddingGraphs,
                                            [trafficSegments[segment]],
                                            self._trafficGenerator,
                                            self._orchestrator.getTelemetry(),
                                            topologyToUse,
                                            "hi_genesis",
                                            f"{len(allRequestsReceived)}_{mutPb}_{indPb}_{crossPb}_{segment}_{i}",
                                            LATENCY,
                                            True,
                                            client,
                                            mutPb = mutPb,
                                            indPb = indPb,
                                            cxPb = crossPb
                                        )
                    else:
                        TUI.appendToSolverLog(
                            f"Running experiment {segment} with default parameters."
                        )

                        for i in range(noOfRuns):
                            solve(
                                allRequestsReceived,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                [trafficSegments[segment]],
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                topologyToUse,
                                "hi_genesis",
                                f"{len(allRequestsReceived)}_0.1_False_10_2_{segment}_{i}",
                                LATENCY,
                                True,
                                client,
                            )

            except Exception as e:
                TUI.appendToSolverLog(str(e), True)

            TUI.appendToSolverLog("Finished experiment.")

    sfcEm: SFCEmulator = SFCEmulator(SFCRGen, HybridSolver, headless)
    sfcEm.startTest(
        topology,
        [trafficSegments[0]],
    )
    sfcEm.end()
