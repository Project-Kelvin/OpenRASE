"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import copy
import json
import os
import random
from time import sleep
import click
import numpy as np
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.hi_genesis import solve
from algorithms.hybrid.genesis import solve as genesisSolve
from mano.orchestrator import Orchestrator
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

def generateSFCRsFromTemplates(sfcrTemplates: list[SFCRequest], segment: int, topo: str, copies: int) -> list[SFCRequest]:
    """
    Generate SFCRs from templates, creating multiple copies for each segment.

    Parameters:
        sfcrTemplates (list[SFCRequest]): The list of SFCR templates.
        segment (int): The current segment index.
        topo (str): The topology type ("mec" or "fat-tree").
        copies (int): The number of copies to generate for each template.

    Returns:
        list[SFCRequest]: A list of generated SFCRs for the current segment.
    """

    allRequests: list[SFCRequest] = []

    if topo == "mec":
        copiesToMake: int = copies
        step: int = 4
        remainder: int = segment % step
        for request in sfcrTemplates[remainder::step]:
            for copyIndex in range(copiesToMake):
                requestCopy: SFCRequest = copy.deepcopy(request)
                requestCopy["sfcrID"] = f"{request['sfcrID']}-{segment}-{copyIndex}"
                allRequests.append(requestCopy)
    elif topo == "fat-tree":
        for request in sfcrTemplates:
            copiesToMake: int = copies if segment == 0 else 1
            for c in range(copiesToMake):
                requestCopy: SFCRequest = copy.deepcopy(request)
                requestCopy["sfcrID"] = f"{request['sfcrID']}-{c}-{segment}"
                allRequests.append(requestCopy)

    return allRequests

@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--client", is_flag=True, default=False, help="Run in client mode.")
@click.option("--env", type=click.Choice(["dc", "milan", "25n50e"]), default="dc", help="Environment to run the algorithm in.")
@click.option("--static-root" , is_flag=True, default=False, help="Run the experiment with a static root individual.")
@click.option("--genesis", is_flag=True, default=False, help="Run the experiment with GENESIS.")
def run(headless: bool, client: bool, env: str, static_root: bool, genesis: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        client (bool): Whether to run the algorithm in client mode.
        env (str): The environment to run the algorithm in.
        static_root (bool): Whether to run the experiment with a static root individual.
        genesis (bool): Whether to run the experiment with GENESIS.

    Returns:
        None
    """

    # Tuned Hyperparameters for HiGENESIS
    indMutPb: float = 0.7
    geneMutPb: float = 0.7
    crossPb: float = 1.0
    dominanceThreshold: float = 0.5

    noOfRuns: int = 20
    absSFCRsToEmbed: int = 70

    if env == "dc":
        noOfCopies: int = 20
        trafficScale: float = 0.1
        cpus: float = 1
        bandwidth: int = 10
        delay: int = 1
        memory: int = 5120
        hostRemoveStartSegment: int = 5

        design: TrafficDesign = generateTrafficDesignFromFile(
            os.path.join(
                f"{getConfig()['repoAbsolutePath']}",
                "src",
                "runs",
                "hybrid",
                "data",
                "requests.csv",
            ),
            trafficScale,
            20,
            False,
        )

        steps: int = len(design)
        segments: int = 10
        stepsPerSegment: int = steps // segments
        trafficSegments: "list[TrafficDesign]" = []
        for segment in range(segments):
            startStep: int = segment * stepsPerSegment
            endStep: int = (segment + 1) * stepsPerSegment

            segmentDesign: TrafficDesign = design[startStep:endStep]
            trafficSegments.append(segmentDesign)

        topology: Topology = generateFatTreeTopology(
            4, bandwidth, cpus, memory, delay
        )

        requests: "list[SFCRequest]" = []
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
            requests = json.load(f)
    else:
        noOfCopies: int = 13
        trafficScale: float = 0.3
        cpus: float = 0.25
        bandwidth: int = 5
        delay: int = 1
        memory: int = 5120

        segmentDuration: int = 30
        design: TrafficDesign = generateTrafficDesignFromIoTTrace(
            os.path.join(
                f"{getConfig()['repoAbsolutePath']}",
                "src",
                "runs",
                "hybrid",
                "data",
                "iot-trace.csv",
            ),
            segmentDuration,
            1000 / trafficScale,
        )
        steps: int = len(design)
        segments: int = steps // (2 * segmentDuration)
        hostRemoveStartSegment: int = int(0.75 * segments)
        stepsPerSegment: int = steps // segments
        trafficSegments: "list[TrafficDesign]" = []

        for segment in range(segments):
            startStep: int = segment * stepsPerSegment
            endStep: int = (segment + 1) * stepsPerSegment

            segmentDesign: TrafficDesign = design[startStep:endStep]
            trafficSegments.append(segmentDesign)

        if env == "milan":
            topology: Topology = generateTopologyFromEdgeList(
                os.path.join(
                    getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "milan.txt"
                ),
                cpus,
                memory,
                bandwidth,
                delay
            )
        else:
            topology: Topology = generateTopologyFromEdgeList(
                os.path.join(
                    getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", "25N50E.txt"
                ),
                cpus,
                memory,
                bandwidth,
                delay
            )

        requests: "list[SFCRequest]" = []
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
            requests = json.load(f)


    class SFCRGen(SFCRequestGenerator):
        """
        Class to generate FG Requests.
        """

        def __init__(self, orchestrator: Orchestrator) -> None:
            """
            Initialize the SFCRGen class.
            """

            super().__init__(orchestrator)
            self.sfcrs: list[SFCRequest] = requests

        def generateRequests(self) -> None:
            """
            Generate the FG Requests.
            """

            sfcrsToSend: "list[SFCRequest]" = []

            for i, sfcr in enumerate(self.sfcrs):
                sfcrToSend: SFCRequest = copy.deepcopy(sfcr)
                sfcrToSend["sfcrID"] = f"sfcr{i}"
                sfcrsToSend.append(sfcrToSend)

            self._orchestrator.sendRequests(sfcrsToSend)

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

            originalRequests: "list[SFCRequest]" = []

            try:
                topologyToUse: Topology = copy.deepcopy(topology)
                while self._requests.empty():
                    pass

                while not self._requests.empty():
                    originalRequests.append(self._requests.get())
                    sleep(0.1)

                for i in range(noOfRuns):
                    removedHosts: "list[int]" = []
                    allRequestsReceived: "list[SFCRequest]" = []

                    for segment in range(segments):
                        for request in originalRequests:
                            copies: int = noOfCopies if segment == 0 else 1
                            for c in range(copies):
                                requestCopy: SFCRequest = copy.deepcopy(request)
                                requestCopy["sfcrID"] = f"{request['sfcrID']}-{c}-{segment}"
                                allRequestsReceived.append(requestCopy)

                        if segment > hostRemoveStartSegment:
                            # Simulate a host failure
                            hosts: list[int] = [i for i in range(len(topologyToUse["hosts"])) if i not in removedHosts]
                            hostIdToRemove: int = random.choice(hosts)
                            hostToRemove: str = f"host{hostIdToRemove}"
                            removedHosts.append(hostIdToRemove)
                            topologyToUse = removeHost(topologyToUse, hostToRemove)
                            TUI.appendToSolverLog(f"Simulated failure of host {hostToRemove}.")

                        self._trafficGenerator.setDesign([trafficSegments[segment]])
                        minimumAR: float = min(absSFCRsToEmbed / len(allRequestsReceived), 1.0)
                        TUI.appendToSolverLog(
                            f"Running experiment {segment} with default parameters. Minimum acceptance rate: {minimumAR}."
                        )

                        if genesis:
                            genesisSolve(
                                allRequestsReceived,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                [trafficSegments[segment]],
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                topology,
                                f"genesis_{env}_{i}",
                                f"{len(allRequestsReceived)}_{trafficScale}_False_{bandwidth}_{cpus}_{segment}",
                                retainPopulation=True,
                                minimumAR=minimumAR
                            )
                        else:
                            solve(
                                allRequestsReceived,
                                self._orchestrator.sendEmbeddingGraphs,
                                self._orchestrator.deleteEmbeddingGraphs,
                                [trafficSegments[segment]],
                                self._trafficGenerator,
                                self._orchestrator.getTelemetry(),
                                topologyToUse,
                                f"hi_genesis_{env}_{i}",
                                f"{len(allRequestsReceived)}_{trafficScale}_False_{bandwidth}_{cpus}_{segment}",
                                LATENCY,
                                retainPopulation=True,
                                isClientMode=client,
                                mutPb=indMutPb,
                                indPb=geneMutPb,
                                cxPb=crossPb,
                                dominanceThreshold=dominanceThreshold,
                                minimumAR=minimumAR,
                                rootIndividual=1 if static_root else -1
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
