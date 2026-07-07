"""
Run the MTDRL SFCR embedding solver in the emulator.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from datetime import datetime
from time import sleep
from timeit import default_timer

import click
import pandas as pd
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig

from algorithms.drl_sfcr_embedding import MTDRLConfig, MTDRLSFCREmbedder
from mano.orchestrator import Orchestrator
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.topology import generateFatTreeTopology
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile
from utils.tui import TUI


def splitTrafficDesign(trafficDesign: TrafficDesign, segments: int) -> list[TrafficDesign]:
    """
    Split one traffic design into approximately equal segments.
    """

    if segments <= 1:
        return [trafficDesign]

    steps: int = len(trafficDesign)
    if steps == 0:
        raise ValueError("Traffic design is empty.")

    stepsPerSegment: int = max(1, math.ceil(steps / segments))
    output: list[TrafficDesign] = []
    for segmentIndex in range(segments):
        startStep = segmentIndex * stepsPerSegment
        if startStep >= steps:
            break
        endStep = min((segmentIndex + 1) * stepsPerSegment, steps)
        output.append(trafficDesign[startStep:endStep])

    return output


def removeHost(topology: Topology, hostID: str) -> Topology:
    """
    Remove a host and all attached links from the topology.
    """

    newTopology: Topology = copy.deepcopy(topology)
    newTopology["hosts"] = [host for host in newTopology["hosts"] if host["id"] != hostID]
    newTopology["links"] = [
        link
        for link in newTopology["links"]
        if link["source"] != hostID and link["destination"] != hostID
    ]
    return newTopology


def toFileName(value: str) -> str:
    """
    Convert string to a filesystem-safe name.
    """

    return (
        value.replace("|", "__")
        .replace("=", "-")
        .replace(".", "_")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(
    headless: bool,
) -> None:
    """
    Run MTDRL-based SFCR embedding experiments.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.

    Returns:
        None
    """

    config = getConfig()
    sfcrPath = os.path.join(
        config["repoAbsolutePath"],
        "src",
        "runs",
        "hybrid",
        "configs",
        "sfcrs.json",
    )
    requestsPath = os.path.join(
        config["repoAbsolutePath"],
        "src",
        "runs",
        "hybrid",
        "data",
        "requests.csv",
    )

    baseTrafficDesign: TrafficDesign = generateTrafficDesignFromFile(
        requestsPath,
        1,
        4,
        False,
        False,
    )
    if len(baseTrafficDesign) == 0:
        raise ValueError("Traffic design is empty.")

    segments: int = 10
    copies: int = 1
    episodes: int = 100
    maxEpisodes: int = 500
    acceptanceThreshold: float = 1.0
    latencyThreshold: float = 150.0
    seed: int = 42
    trafficSegments: list[TrafficDesign] = (
        splitTrafficDesign(baseTrafficDesign, segments)
    )
    topology: Topology = generateFatTreeTopology(4, 10, 1, 5120, 10)
    vnfCatalog: list[str] = config["vnfs"]["names"]
    experimentName: str = (
        f"RL_0.1_False_10_2"
    )
    artifactsDir: str = os.path.join(
        config["repoAbsolutePath"],
        "artifacts",
        "drl_sfcr",
    )
    if not os.path.exists(artifactsDir):
        os.makedirs(artifactsDir)
    metricsPath: str = os.path.join(artifactsDir, "experiments.csv")
    summaryLogPath: str = os.path.join(artifactsDir, "experiments.log")
    experimentLogPath: str = os.path.join(artifactsDir, f"{toFileName(experimentName)}.log")
    if not os.path.exists(metricsPath):
        with open(metricsPath, "w", encoding="utf8") as metricsFile:
            metricsFile.write(
                "experiment,segment,acceptance_ratio,calculated_delay,measured_latency,total_execution_time,episodes_used,converged,termination_reason\n"
            )

    class SFCRGen(SFCRequestGenerator):
        """
        SFCR generator for MTDRL run.
        """

        def __init__(self, orchestrator: Orchestrator) -> None:
            super().__init__(orchestrator)
            with open(sfcrPath, "r", encoding="utf8") as sfcrFile:
                self._sfcrTemplates: list[SFCRequest] = json.load(sfcrFile)

        def generateRequests(self) -> None:
            requests: list[SFCRequest] = []
            for index, template in enumerate(self._sfcrTemplates):
                sfcr: SFCRequest = copy.deepcopy(template)
                sfcr["sfcrID"] = f"sfcr{index}"
                sfcr.setdefault("latency", 200)
                sfcr.setdefault("strictOrder", [])
                requests.append(sfcr)
            self._orchestrator.sendRequests(requests)

    class DRLSolver(Solver):
        """
        Solver backed by MTDRL SFCR embedder.
        """

        @staticmethod
        def _getLinkDelay(topology: Topology, source: str, destination: str) -> float:
            links = [
                topoLink
                for topoLink in topology["links"]
                if (
                    topoLink["source"] == source and topoLink["destination"] == destination
                ) or (
                    topoLink["source"] == destination and topoLink["destination"] == source
                )
            ]
            if len(links) == 0:
                return 0.0
            return float(links[0].get("delay", 0.0) or 0.0)

        @classmethod
        def _calculateDelay(cls, topology: Topology, embeddings: list[EmbeddingGraph]) -> float:
            if len(embeddings) == 0:
                return float("nan")

            totalDelay: float = 0.0
            for eg in embeddings:
                egDelay: float = 0.0
                for link in eg["links"]:
                    path: list[str] = [link["source"]["id"]]
                    path.extend(link["links"])
                    path.append(link["destination"]["id"])
                    for index in range(len(path) - 1):
                        egDelay += cls._getLinkDelay(topology, path[index], path[index + 1])
                totalDelay += egDelay
            return totalDelay / len(embeddings)

        @staticmethod
        def _calculateMeasuredLatency(trafficData: pd.DataFrame) -> float:
            if (
                trafficData.empty
                or "_time" not in trafficData.columns
                or "_value" not in trafficData.columns
                or "sfcID" not in trafficData.columns
            ):
                return float("nan")

            data = trafficData.copy()
            data["_time"] = data["_time"] // 1000000000
            groupedTrafficData: pd.DataFrame = data.groupby(["_time", "sfcID"]).agg(
                reqps=("_value", "count"),
                medianLatency=("_value", "median"),
            )
            return float(groupedTrafficData["medianLatency"].mean())

        @staticmethod
        def _buildIngressTrafficMap(requests: list[SFCRequest], trafficDesign: TrafficDesign) -> dict[str, float]:
            ingressTargets: list[float] = [
                float(slot["target"])
                for slot in trafficDesign
                if "target" in slot and slot["target"] is not None
            ]
            if len(ingressTargets) == 0:
                raise ValueError("Traffic segment has no ingress target values.")
            segmentIngress: float = max(ingressTargets)
            return {request["sfcrID"]: segmentIngress for request in requests}

        @staticmethod
        def _writeMetrics(
            segment: int,
            acceptanceRatio: float,
            calculatedDelay: float,
            measuredLatency: float,
            totalExecutionTime: float,
            episodesUsed: int,
            converged: bool,
            terminationReason: str,
            acceptedCount: int,
            failedCount: int,
            topologyUsed: Topology,
        ) -> None:
            with open(metricsPath, "a", encoding="utf8") as metricsFile:
                metricsFile.write(
                    f"{experimentName},{segment},{acceptanceRatio},{calculatedDelay},"
                    f"{measuredLatency},{totalExecutionTime},{episodesUsed},"
                    f"{str(converged).lower()},{terminationReason}\n"
                )
            timestamp: str = datetime.now().isoformat()
            logLine: str = (
                f"[{timestamp}] experiment={experimentName} segment={segment} "
                f"acceptance_ratio={acceptanceRatio} calculated_delay={calculatedDelay} "
                f"measured_latency={measuredLatency} total_execution_time={totalExecutionTime} "
                f"episodes_used={episodesUsed} converged={str(converged).lower()} "
                f"termination_reason={terminationReason}"
            )
            with open(summaryLogPath, "a", encoding="utf8") as summaryLogFile:
                summaryLogFile.write(f"{logLine}\n")
            with open(experimentLogPath, "a", encoding="utf8") as experimentLogFile:
                experimentLogFile.write(
                    f"{logLine}\n"
                    f"accepted_sfcrs={acceptedCount} failed_sfcrs={failedCount} "
                    f"hosts_available={len(topologyUsed['hosts'])}\n"
                )

        def _runSegment(
            self,
            requests: list[SFCRequest],
            topologyToUse: Topology,
            segmentDesign: TrafficDesign,
            segment: int,
        ) -> None:
            startTime: float = default_timer()
            episodeStep: int = max(1, min(episodes, maxEpisodes))
            drlConfig = MTDRLConfig(trainingEpisodes=episodeStep)
            ingressTrafficMap: dict[str, float] = DRLSolver._buildIngressTrafficMap(requests, segmentDesign)
            embedder = MTDRLSFCREmbedder(
                topology=topologyToUse,
                vnfCatalog=vnfCatalog,
                config=drlConfig,
                seed=seed + segment,
            )
            episodesUsed: int = 0
            converged: bool = False
            terminationReason: str = "max_episodes_reached"
            acceptanceRatio: float = 0.0
            calculatedDelay: float = float("nan")
            measuredLatency: float = float("nan")
            accepted: list[EmbeddingGraph] = []
            failed: list[SFCRequest] = []
            trafficDuration: int = calculateTrafficDuration(segmentDesign)

            while episodesUsed < maxEpisodes:
                episodesThisRound: int = min(episodeStep, maxEpisodes - episodesUsed)
                embedder._config.trainingEpisodes = episodesThisRound
                accepted, failed = embedder.embed(requests, ingressTrafficMap=ingressTrafficMap)
                episodesUsed += episodesThisRound

                acceptanceRatio = (
                    float(len(accepted)) / float(len(requests)) if len(requests) > 0 else 0.0
                )
                calculatedDelay = DRLSolver._calculateDelay(topologyToUse, accepted)
                measuredLatency = float("nan")

                if len(accepted) > 0:
                    self._trafficGenerator.setDesign([segmentDesign])
                    self._orchestrator.sendEmbeddingGraphs(accepted)
                    try:
                        TUI.appendToSolverLog(
                            f"Segment {segment}: waiting for {trafficDuration}s after {episodesUsed} episodes."
                        )
                        sleep(trafficDuration)
                        trafficData: pd.DataFrame = self._trafficGenerator.getData(f"{trafficDuration:.0f}s")
                        measuredLatency = DRLSolver._calculateMeasuredLatency(trafficData)
                    finally:
                        self._orchestrator.deleteEmbeddingGraphs(accepted)

                hasLatency: bool = not math.isnan(measuredLatency)
                converged = (
                    acceptanceRatio >= acceptanceThreshold
                    and hasLatency
                    and measuredLatency <= latencyThreshold
                )
                if converged:
                    terminationReason = "converged"
                    break

            totalExecutionTime: float = default_timer() - startTime
            DRLSolver._writeMetrics(
                segment,
                acceptanceRatio,
                calculatedDelay,
                measuredLatency,
                totalExecutionTime,
                episodesUsed,
                converged,
                terminationReason,
                len(accepted),
                len(failed),
                topologyToUse,
            )
            TUI.appendToSolverLog(
                f"Segment {segment}: AR={acceptanceRatio:.4f}, "
                f"CalcDelay={calculatedDelay}, Latency={measuredLatency}, "
                f"Exec={totalExecutionTime:.4f}s, Episodes={episodesUsed}, "
                f"Converged={str(converged).lower()}"
            )

        def generateEmbeddingGraphs(self) -> None:
            try:
                while self._requests.empty():
                    sleep(0.05)

                originalRequests: list[SFCRequest] = []
                while not self._requests.empty():
                    originalRequests.append(self._requests.get())
                    sleep(0.05)

                topologyToUse: Topology = copy.deepcopy(self._orchestrator.getTopology())
                allRequests: list[SFCRequest] = []
                removedHosts: list[str] = []
                randomizer = random.Random(seed)
                failureStartSegment: int = max(1, len(trafficSegments) // 2)

                for segment, segmentDesign in enumerate(trafficSegments):
                    copiesThisSegment: int = copies if segment == 0 else 1
                    segmentRequests: list[SFCRequest] = []
                    for request in originalRequests:
                        for copyIndex in range(copiesThisSegment):
                            requestCopy: SFCRequest = copy.deepcopy(request)
                            requestCopy["sfcrID"] = f"{request['sfcrID']}-{copyIndex}-{segment}"
                            segmentRequests.append(requestCopy)
                    allRequests.extend(segmentRequests)

                    if segment >= failureStartSegment and len(topologyToUse["hosts"]) > 1:
                        remainingHosts: list[str] = [
                            host["id"] for host in topologyToUse["hosts"] if host["id"] not in removedHosts
                        ]
                        if len(remainingHosts) > 0:
                            hostToRemove: str = randomizer.choice(remainingHosts)
                            removedHosts.append(hostToRemove)
                            topologyToUse = removeHost(topologyToUse, hostToRemove)
                            TUI.appendToSolverLog(
                                f"Segment {segment}: simulated failure of host {hostToRemove}."
                            )

                    self._runSegment(allRequests, topologyToUse, segmentDesign, segment)
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)
            TUI.appendToSolverLog("Finished MTDRL solver run.")

    sfcEmulator = SFCEmulator(SFCRGen, DRLSolver, headless)
    sfcEmulator.startTest(topology, [trafficSegments[0]])
    sfcEmulator.end()
