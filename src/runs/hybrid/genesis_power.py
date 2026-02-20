"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import copy
import json
import os
import random
from time import sleep
from typing import Any
import click
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.hybrid.constants.gensis_objective import POWER
from algorithms.hybrid.genesis import solve
from mano.orchestrator import Orchestrator
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology, generateTopologyFromEdgeList
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile, generateTrafficDesignFromIoTTrace
from utils.tui import TUI


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.

    Returns:
        None
    """

    mecTopos: list[str] = ["milan", "25N50E"]

    for mecTopo in mecTopos:

        segmentDuration: int = 1 * 60
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
            1000,
        )

        steps: int = len(design)
        segments: int = steps // segmentDuration
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

        topology: Topology = generateTopologyFromEdgeList(
            os.path.join(
                getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", f"{mecTopo}.txt"
            ),
            1,
            5 * 1024,
            10,
            10
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

            newTopology["hosts"] = [
                host for host in newTopology["hosts"] if host["id"] != hostID
            ]
            newTopology["links"] = [
                link
                for link in newTopology["links"]
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
                COPIES: int = 1
                try:
                    topologyToUse: Topology = copy.deepcopy(topology)
                    while self._requests.empty():
                        pass

                    while not self._requests.empty():
                        originalRequests.append(self._requests.get())
                        sleep(0.1)

                    for segment in range(segments):
                        remainder: int = segment % 2
                        for request in originalRequests[remainder::2]:
                            for copyIndex in range(COPIES):
                                requestCopy: SFCRequest = copy.deepcopy(request)
                                requestCopy["sfcrID"] = f"{request['sfcrID']}-{segment}-{copyIndex}"
                                allRequestsReceived.append(requestCopy)

                        if segment > int(segments * 0.75):
                            # Simulate a host failure
                            hosts: list[int] = [
                                i
                                for i in range(len(topologyToUse["hosts"]))
                                if i not in removedHosts
                            ]
                            hostIdToRemove: int = random.choice(hosts)
                            hostToRemove: str = f"host{hostIdToRemove}"
                            removedHosts.append(hostIdToRemove)
                            topologyToUse = removeHost(topologyToUse, hostToRemove)
                            TUI.appendToSolverLog(
                                f"Simulated failure of host {hostToRemove}."
                            )

                        self._trafficGenerator.setDesign([trafficSegments[segment]])
                        solve(
                            allRequestsReceived,
                            self._orchestrator.sendEmbeddingGraphs,
                            self._orchestrator.deleteEmbeddingGraphs,
                            [trafficSegments[segment]],
                            self._trafficGenerator,
                            self._orchestrator.getTelemetry(),
                            topologyToUse,
                            "genesis_power",
                            f"{len(allRequestsReceived)}_0.1_False_10_1_{segment}_{mecTopo}",
                            POWER,
                            retainPopulation=True
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
