"""
Run the MTDRL SFCR embedding solver in the emulator.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from datetime import datetime
from typing import Any, Union, cast
from time import sleep
from timeit import default_timer

import click
import pandas as pd
from algorithms.hybrid.utils.demand_predictions import DemandPredictions
from algorithms.hybrid.models.traffic import TimeSFCRequests
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
from utils.embedding_graph import traverseVNF
from utils.topology import generateFatTreeTopology, generateTopologyFromEdgeList
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile, generateTrafficDesignFromIoTTrace
from utils.tui import TUI


def splitTrafficDesign(trafficDesign: TrafficDesign, segments: int, static: bool) -> list[TrafficDesign]:
    """
    Split one traffic design into approximately equal segments.

    Parameters:
        trafficDesign (TrafficDesign): The original traffic design to split.
        segments (int): The number of segments to split the traffic design into.
        static (bool): Whether to use static splitting (equal number of steps per segment) or

    Returns:
        list[TrafficDesign]: A list of traffic design segments.
    """

    if static:
        return [trafficDesign]

    steps: int = len(trafficDesign)
    stepsPerSegment: int = steps // segments
    trafficSegments: "list[TrafficDesign]" = []
    for segment in range(segments):
        startStep: int = segment * stepsPerSegment
        endStep: int = (segment + 1) * stepsPerSegment

        segmentDesign: TrafficDesign = trafficDesign[startStep:endStep]
        trafficSegments.append(segmentDesign)

    return trafficSegments

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
        copies: int = 1
        step: int = 4
        remainder: int = segment % step
        for request in sfcrTemplates[remainder::step]:
            for copyIndex in range(copies):
                requestCopy: SFCRequest = copy.deepcopy(request)
                requestCopy["sfcrID"] = f"{request['sfcrID']}-{segment}-{copyIndex}"
                allRequests.append(requestCopy)
    elif topo == "fat-tree":
        for request in sfcrTemplates:
            copies: int = copies if segment == 0 else 1
            for c in range(copies):
                requestCopy: SFCRequest = copy.deepcopy(request)
                requestCopy["sfcrID"] = f"{request['sfcrID']}-{c}-{segment}"
                allRequests.append(requestCopy)

    return allRequests
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


REQUEST_SIZE_MBPS: float = 0.05
HOST_VIRTUALISATION_DELAY: float = 1.0
OVERLOAD_DELAY_PENALTY: float = 1_000_000.0
MAX_CALCULATED_DELAY: float = 1_000_000_000.0


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
@click.option("--topology", type=str, default="fat-tree", help="Topology to use for the experiment.")
@click.option("--static", is_flag=True, default=False, help="Run static embedding instead of MTDRL.")
def run(
    headless: bool,
    topology: str,
    static: bool
) -> None:
    """
    Run MTDRL-based SFCR embedding experiments.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
        topology (str): Topology to use for the experiment.

    Returns:
        None
    """

    # Ensure redirected logs are flushed immediately in headless/background runs.
    stdout: Any = sys.stdout
    stderr: Any = sys.stderr
    if hasattr(stdout, "reconfigure"):
        stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(stderr, "reconfigure"):
        stderr.reconfigure(line_buffering=True, write_through=True)

    topos: list[str] = []
    experimentConfig: list[tuple[int, float, bool, int, int]] = []

    if static:
        topology = "fat-tree"
        experimentConfig = [
            (8, 0.1, False, 10, 2),
            (8, 0.2, False, 10, 2),
            (8, 0.1, True, 10, 2),
            (8, 0.1, False, 5, 2),
            (8, 0.1, False, 10, 1),
        ]
    else:
        experimentConfig = [
            (20, 0.1, False, 10, 2),
        ]

    print(f"Running MTDRL SFCR embedding experiments with topology '{topology}'...")

    if topology == "mec":
        topos =["milan", "25N50E"]
    else:
        topos = ["fat-tree"]

    for expConfig in experimentConfig:
        for topoName in topos:
            config = getConfig()
            experimentName: str = (
                f"RL_{expConfig[1]}_{expConfig[2]}_{expConfig[3]}_{expConfig[4]}_{topoName}"
            )
            artifactsDir: str = os.path.join(
                config["repoAbsolutePath"],
                "artifacts"
            )
            if not os.path.exists(artifactsDir):
                os.makedirs(artifactsDir)

            experimentLogDir: str = os.path.join(
                artifactsDir,
                "experiments"
            )
            if not os.path.exists(experimentLogDir):
                os.makedirs(experimentLogDir)

            artifactsDir: str = os.path.join(
                experimentLogDir,
                f"rl_dc_{static}",
            )
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
                expConfig[0] * 10,
                4 if static else 20,
                False,
                expConfig[2],
            )

            topo: Topology = generateFatTreeTopology(4, expConfig[3], expConfig[4], 5120, 10)

            segments: int = 10

            failureStartSegment: int = 5

            if topology == "mec":
                topo = generateTopologyFromEdgeList(
                    os.path.join(
                        getConfig()["repoAbsolutePath"], "src", "runs", "hybrid", "data", f"{topoName}.txt"
                    ),
                    1,
                    5 * 1024,
                    10,
                    1
                )

                segmentDuration: int = 60
                baseTrafficDesign = generateTrafficDesignFromIoTTrace(
                    os.path.join(
                        f"{getConfig()['repoAbsolutePath']}",
                        "src",
                        "runs",
                        "hybrid",
                        "data",
                        "iot-trace.csv",
                    ),
                    segmentDuration,
                    10000,
                )

                sfcrPath = os.path.join(
                    getConfig()["repoAbsolutePath"],
                    "src",
                    "runs",
                    "hybrid",
                    "configs",
                    "sfcrs_random.json",
                )

                segments = len(baseTrafficDesign) // (2 * segmentDuration)

                artifactsDir: str = os.path.join(
                    experimentLogDir,
                    f"rl_mec_{topoName}",
                )

                failureStartSegment = int(segments * 0.75)
                copies = 1

            if len(baseTrafficDesign) == 0:
                raise ValueError("Traffic design is empty.")

            maxEpisodes: int = 200 if not static else 500
            acceptanceThreshold: float = 1.0
            latencyThreshold: float = 150.0
            seed: int = 42
            trafficSegments: list[TrafficDesign] = (
                splitTrafficDesign(baseTrafficDesign, segments, static)
            )

            vnfCatalog: list[str] = config["vnfs"]["names"]

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
                    requestPayload: list[Union[SFCRequest, EmbeddingGraph]] = cast(
                        list[Union[SFCRequest, EmbeddingGraph]],
                        requests,
                    )
                    TUI.appendToSolverLog(f"Generated {len(requests)} SFCRs to send.")
                    self._orchestrator.sendRequests(requestPayload)

            class DRLSolver(Solver):
                """
                Solver backed by MTDRL SFCR embedder.
                """

                print("Initialising Demand Predictions...")
                _demandPredictions: DemandPredictions = DemandPredictions()

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

                @staticmethod
                def _getLinkBandwidth(topology: Topology, source: str, destination: str) -> float:
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
                    return float(links[0].get("bandwidth", 0.0) or 0.0)

                @staticmethod
                def _getHostCPU(topology: Topology, hostID: str) -> float:
                    hosts = [host for host in topology["hosts"] if host["id"] == hostID]
                    if len(hosts) == 0:
                        return 0.0
                    return float(hosts[0].get("cpu", 0.0) or 0.0)

                @staticmethod
                def _getEmbeddedVNFs(embedding: EmbeddingGraph) -> list[tuple[str, str, int]]:
                    embeddedVNFs: list[tuple[str, str, int]] = []

                    def parseVNF(vnf: dict, depth: int) -> None:
                        if "vnf" not in vnf or "host" not in vnf:
                            return
                        if "id" not in vnf["vnf"] or "id" not in vnf["host"]:
                            return
                        embeddedVNFs.append((vnf["vnf"]["id"], vnf["host"]["id"], depth))

                    traverseVNF(embedding["vnfs"], parseVNF)
                    return embeddedVNFs

                @classmethod
                def _calculatePropagationDelay(cls, topology: Topology, embedding: EmbeddingGraph) -> float:
                    propagationDelay: float = 0.0
                    for forwardingLink in embedding["links"]:
                        path: list[str] = [forwardingLink["source"]["id"]]
                        path.extend(forwardingLink["links"])
                        path.append(forwardingLink["destination"]["id"])
                        for index in range(len(path) - 1):
                            propagationDelay += cls._getLinkDelay(topology, path[index], path[index + 1])
                    return propagationDelay

                @classmethod
                def _calculateQueueDelay(
                    cls,
                    topology: Topology,
                    embedding: EmbeddingGraph,
                    ingressReqps: float,
                ) -> float:
                    queueDelay: float = 0.0
                    for forwardingLink in embedding["links"]:
                        divisor: int = int(forwardingLink.get("divisor", 1) or 1)
                        effectiveReqps: float = ingressReqps / float(max(1, divisor))
                        demandSizeMbps: float = effectiveReqps * REQUEST_SIZE_MBPS
                        path: list[str] = [forwardingLink["source"]["id"]]
                        path.extend(forwardingLink["links"])
                        path.append(forwardingLink["destination"]["id"])
                        for index in range(len(path) - 1):
                            bandwidth: float = cls._getLinkBandwidth(topology, path[index], path[index + 1])
                            residualBandwidth: float = bandwidth - demandSizeMbps
                            if residualBandwidth <= 0:
                                queueDelay += OVERLOAD_DELAY_PENALTY
                                continue
                            queueDelay += 1.0 / residualBandwidth
                    return queueDelay

                @classmethod
                def _calculateProcessingDelay(
                    cls,
                    topology: Topology,
                    embedding: EmbeddingGraph,
                    ingressReqps: float,
                ) -> float:
                    processingDelay: float = 0.0
                    for vnfID, hostID, depth in cls._getEmbeddedVNFs(embedding):
                        divisor: int = 2 ** max(0, depth - 1)
                        effectiveReqps: float = ingressReqps / float(max(1, divisor))
                        cpuDemand: float = cls._demandPredictions.getDemand(vnfID, effectiveReqps)["cpu"]
                        cpuAvailable: float = cls._getHostCPU(topology, hostID)
                        residualCPU: float = cpuAvailable - cpuDemand
                        if residualCPU <= 0:
                            processingDelay += OVERLOAD_DELAY_PENALTY
                            continue
                        processingDelay += 1.0 / residualCPU
                    return processingDelay

                @classmethod
                def _calculateVirtualisationDelay(cls, embedding: EmbeddingGraph) -> float:
                    return HOST_VIRTUALISATION_DELAY * float(len(cls._getEmbeddedVNFs(embedding)))

                @classmethod
                def _calculateDelay(
                    cls,
                    topology: Topology,
                    embeddings: list[EmbeddingGraph],
                    ingressTrafficMap: dict[str, float],
                ) -> float:
                    if len(embeddings) == 0:
                        return float("nan")

                    trafficSnapshot: dict[str, float] = {
                        embedding["sfcID"]: float(ingressTrafficMap.get(embedding["sfcID"], 0.0))
                        for embedding in embeddings
                    }
                    trafficSeries: TimeSFCRequests = cast(TimeSFCRequests, [trafficSnapshot])
                    cls._demandPredictions.cacheResourceDemands(embeddings, trafficSeries)

                    totalLatency: float = 0.0
                    for embedding in embeddings:
                        ingressReqps: float = float(ingressTrafficMap.get(embedding["sfcID"], 0.0))
                        propagationDelay: float = cls._calculatePropagationDelay(topology, embedding)
                        queueDelay: float = cls._calculateQueueDelay(topology, embedding, ingressReqps)
                        processingDelay: float = cls._calculateProcessingDelay(topology, embedding, ingressReqps)
                        virtualisationDelay: float = cls._calculateVirtualisationDelay(embedding)
                        totalLatency += (
                            propagationDelay
                            + queueDelay
                            + processingDelay
                            + virtualisationDelay
                        )

                    averageLatency: float = totalLatency / float(len(embeddings))
                    if not math.isfinite(averageLatency):
                        return MAX_CALCULATED_DELAY
                    return averageLatency

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
                    episodeStep: int = maxEpisodes
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

                    embedder._config.trainingEpisodes = episodeStep
                    accepted, failed = embedder.embed(requests, ingressTrafficMap=ingressTrafficMap)
                    episodesUsed = episodeStep

                    acceptanceRatio = (
                        float(len(accepted)) / float(len(requests)) if len(requests) > 0 else 0.0
                    )
                    calculatedDelay = DRLSolver._calculateDelay(
                        topologyToUse,
                        accepted,
                        ingressTrafficMap,
                    )
                    if not math.isfinite(calculatedDelay):
                        calculatedDelay = MAX_CALCULATED_DELAY

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
                    terminationReason = "converged" if converged else "max_episodes_reached"

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
                    TUI.appendToSolverLog(f"Generating embedding graphs for topology '{topoName}'...")
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

                        TUI.appendToSolverLog(f"Starting segment processing for topology '{topoName}'...")
                        for segment, segmentDesign in enumerate(trafficSegments):
                            allRequests = generateSFCRsFromTemplates(originalRequests, segment, topology, expConfig[0])

                            if segment > failureStartSegment and len(topologyToUse["hosts"]) > 1:
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

            print(f"Starting MTDRL solver run for topology '{topoName}'...")
            sfcEmulator = SFCEmulator(SFCRGen, DRLSolver, headless)
            try:
                print(f"Starting test for topology '{topoName}'...")
                sfcEmulator.startTest(topo, [trafficSegments[0]])
            finally:
                TUI.appendToSolverLog("Ending SFC emulator.")
                sfcEmulator.end()
