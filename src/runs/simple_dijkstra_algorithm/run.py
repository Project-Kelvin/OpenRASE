"""
This runs the Simple Dijsktra Algorithm.
"""

import copy
import json
from time import sleep
from timeit import default_timer
from typing import Union
from mano.telemetry import Telemetry
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from shared.models.sfc_request import SFCRequest
from shared.models.config import Config
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from mano.orchestrator import Orchestrator
from models.calibrate import ResourceDemand
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesignFromFile, getTrafficDesignRate
import click
from utils.tui import TUI
import os

config: Config = getConfig()
configPath: str = (
    f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"
)


directory = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/simple_dijkstra_algorithm"
)

if not os.path.exists(directory):
    os.makedirs(directory)

topology1: Topology = generateFatTreeTopology(4, 1000, 4, None)
topologyPointFive: Topology = generateFatTreeTopology(4, 1000, 2, None)
logFilePath: str = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/simple_dijkstra_algorithm/experiments.csv"
)
hostDataFilePath: str = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/simple_dijkstra_algorithm/host_data.csv"
)
latencyDataFilePath: str = (
    f"{config['repoAbsolutePath']}/artifacts/experiments/simple_dijkstra_algorithm/latency_data.csv"
)

with open(logFilePath, "w", encoding="utf8") as logFile:
    logFile.write("experiment,failed_fgs,accepted_fgs,execution_time,deployment_time\n")

with open(hostDataFilePath, "w", encoding="utf8") as hostDataFile:
    hostDataFile.write("experiment,host,cpu,memory,duration\n")

with open(latencyDataFilePath, "w", encoding="utf8") as latencyData:
    latencyData.write("experiment,sfc,requests,average_latency,duration\n")

experimentID: str = ""
experimentName: str = ""

def appendToLog(message: str) -> None:
    """
    Append to the log.

    Parameters:
        message (str): The message to append.
    """

    with open(logFilePath, "a", encoding="utf8") as log:
        log.write(f"{message}\n")


with open(f"{configPath}/traffic-design.json", "r", encoding="utf8") as trafficFile:
    trafficDesign: "list[TrafficDesign]" = [json.load(trafficFile)]


class FGR(FGRequestGenerator):
    """
    FG Request Generator.
    """

    def __init__(self, orchestrator: Orchestrator) -> None:
        super().__init__(orchestrator)
        self._fgs: "list[EmbeddingGraph]" = []

        with open(
            f"{configPath}/forwarding-graphs.json", "r", encoding="utf8"
        ) as fgFile:
            fgs: "list[EmbeddingGraph]" = json.load(fgFile)
            for fg in fgs:
                self._fgs.append(copy.deepcopy(fg))


class FGR4SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs once.
    """

    def generateRequests(self) -> None:
        for index, fg in enumerate(self._fgs):
            fg["sfcrID"] = f"sfc{index}-{experimentName}"

        self._orchestrator.sendRequests(self._fgs)


class FGR8SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs twice.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            for i in range(0, 2):
                copiedFG: EmbeddingGraph = copy.deepcopy(fg)
                copiedFG["sfcrID"] = f"sfc{index}-{i}-{experimentName}"
                copiedFGs.append(copiedFG)

        self._fgs = copiedFGs

        self._orchestrator.sendRequests(self._fgs)


class FGR32SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs 8 times.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            for i in range(0, 8):
                copiedFG: EmbeddingGraph = copy.deepcopy(fg)
                copiedFG["sfcrID"] = f"sfc{index}-{i}-{experimentName}"
                copiedFGs.append(copiedFG)

        self._fgs = copiedFGs

        self._orchestrator.sendRequests(self._fgs)


class FGR16SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs four times.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            for i in range(0, 4):
                copiedFG: EmbeddingGraph = copy.deepcopy(fg)
                copiedFG["sfcrID"] = f"sfc{index}-{i}-{experimentName}"
                copiedFGs.append(copiedFG)

        self._fgs = copiedFGs

        self._orchestrator.sendRequests(self._fgs)


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def __init__(
        self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator
    ) -> None:
        super().__init__(orchestrator, trafficGenerator)
        self._resourceDemands: "dict[str, ResourceDemand]" = None
        self._topology: Topology = None

    def generateEmbeddingGraphs(self) -> None:
        try:
            while self._requests.empty():
                pass
            requests: "list[Union[FGR, SFCRequest]]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)
            self._topology: Topology = self._orchestrator.getTopology()
            trafficDesignPath = f"{configPath}/traffic-design.json"
            with open(trafficDesignPath, "r", encoding="utf8") as traffic:
                design = json.load(traffic)
            maxTarget: int = max(design, key=lambda x: x["target"])["target"]
            sda = SimpleDijkstraAlgorithm(
                requests, self._topology, maxTarget
            )
            start: float = default_timer()
            fgs, failedFGs, _nodes = sda.run()
            end: float = default_timer()
            executionTime = end - start

            logRow: "list[str]" = []
            logRow.append(experimentID)
            TUI.appendToSolverLog(f"Failed FGs: {len(failedFGs)}")
            TUI.appendToSolverLog(f"Accepted FGs: {len(fgs)}")
            logRow.append(str(len(failedFGs)))
            logRow.append(str(len(fgs)))
            TUI.appendToSolverLog(
                f"Acceptance Ratio: {len(fgs) / (len(fgs) + len(failedFGs)) * 100:.2f}%"
            )
            logRow.append(f"{executionTime:.6f}")
            TUI.appendToSolverLog(f"Execution Time: {executionTime:.6f}s")

            TUI.appendToSolverLog("Deploying Embedding Graphs.")
            start = default_timer()
            self._orchestrator.sendEmbeddingGraphs(fgs)
            end = default_timer()
            TUI.appendToSolverLog("Finished deploying Embedding Graphs.")
            deploymentTime = end - start

            logRow.append(f"{deploymentTime:.6f}")
            TUI.appendToSolverLog("Deployment Time: {deploymentTime:.6f}s")

            with open(logFilePath, "a", encoding="utf8") as log:
                log.write(f"{','.join(logRow)}\n")

            trafficDuration: int = calculateTrafficDuration(
                self._trafficGenerator.getDesign()[0]
            )
            TUI.appendToSolverLog(f"Waiting for {trafficDuration}s.")
            time: int = 0
            telemetry: Telemetry = self._orchestrator.getTelemetry()

            try:
                while time < trafficDuration:
                    start: float = default_timer()
                    hostData: HostData = telemetry.getHostData()
                    end: float = default_timer()
                    duration: int = round(end - start, 0)
                    trafficData: "dict[str, TrafficData]" = (
                        self._trafficGenerator.getData(f"{duration:.0f}s")
                    )

                    for key, data in hostData.items():
                        hostRow: "list[str]" = []
                        hostRow.append(experimentID)
                        hostRow.append(key)

                        hostRow.append(str(data["cpuUsage"][0]))
                        hostRow.append(
                            str(
                                data["memoryUsage"][0] / (1024 * 1024)
                                if data["memoryUsage"][0] != 0
                                else 0
                            )
                        )
                        hostRow.append(str(duration))
                        with open(
                            hostDataFilePath, "a", encoding="utf8"
                        ) as hostDataLogFile:
                            hostDataLogFile.write(f"{','.join(hostRow)}\n")

                    for key, data in trafficData.items():
                        row: "list[str]" = []
                        row.append(experimentID)
                        row.append(key)
                        row.append(str(data["httpReqs"]))
                        row.append(str(data["averageLatency"]))
                        row.append(str(duration))
                        TUI.appendToSolverLog(f"{key}: {str(data['averageLatency'])}")
                        with open(
                            latencyDataFilePath, "a", encoding="utf8"
                        ) as latencyDataFile:
                            latencyDataFile.write(f"{','.join(row)}\n")

                    time += duration
            except Exception as e:
                TUI.appendToSolverLog(str(e), True)

            TUI.appendToSolverLog("Finished waiting.")

            data: "dict[str, TrafficData]" = self._trafficGenerator.getData(
                f"{trafficDuration}s"
            )

            TUI.appendToSolverLog("Finished experiment.")
            sleep(2)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)
        TUI.exit()


def getTrafficDesign() -> None:
    """
    Get the Traffic Design.
    """

    design: TrafficDesign = generateTrafficDesignFromFile(
        f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/requests.csv",
        0.20,
        1,
        True
    )

    with open(f"{configPath}/traffic-design.json", "w", encoding="utf8") as traffic:
        json.dump(design, traffic, indent=4)

    print(getTrafficDesignRate(design, [2]*(len(design) // 2)))


@click.command()
@click.option("--experiment", type=int, help="The experiment to run.")
def run(experiment: int) -> None:
    """
    Run the Simple Dijkstra Algorithm.

    Parameters:
        experiment (int): The experiment to run.
    """

    def runExperiment(fgr: FGR, topology: Topology) -> None:
        """
        Run an experiment.

        Parameters:
            fgr (FGR): The FG Request Generator.
            topology (Topology): The topology.
        """

        nonlocal experiment
        global experimentName
        sfcEm: SFCEmulator = SFCEmulator(fgr, SFCSolver)
        sfcEm.startTest(topology, trafficDesign)
        sfcEm.end()
        experiment = ""
        experimentName = ""

    def experiment1() -> None:
        """
        Run Experiment 1.
        """

        # Experiment 1 - 4 SFCs 0.5
        nonlocal experiment
        global experimentName
        experimentName = "one"
        experiment = "SFCs:4-Topology:0.5"
        runExperiment(FGR4SFC, topologyPointFive)

    def experiment2() -> None:
        """
        Run Experiment 2.
        """

        # Experiment 2 - 8 SFCs 0.5
        nonlocal experiment
        global experimentName
        experimentName = "two"
        experiment = "SFCs:8-Topology:0.5"
        runExperiment(FGR8SFC, topologyPointFive)

    def experiment3() -> None:
        """
        Run Experiment 3.
        """

        # Experiment 3 - 32 SFCs 0.5
        nonlocal experiment
        global experimentName
        experimentName = "three"
        experiment = "SFCs:32-Topology:0.5"
        runExperiment(FGR32SFC, topologyPointFive)

    def experiment4() -> None:
        """
        Run Experiment 4.
        """

        # Experiment 4 - 16 SFCs 0.5
        nonlocal experiment
        global experimentName
        experimentName = "four"
        experiment = "SFCs:16-Topology:0.5"
        runExperiment(FGR16SFC, topologyPointFive)

    def experiment5() -> None:
        """
        Run Experiment 5.
        """

        # Experiment 5 - 4 SFCs 1
        nonlocal experiment
        global experimentName
        experimentName = "five"
        experiment = "SFCs:4-Topology:1"
        runExperiment(FGR4SFC, topology1)

    def experiment6() -> None:
        """
        Run Experiment 6.
        """

        # Experiment 6 - 8 SFCs 1
        nonlocal experiment
        global experimentName
        experimentName = "six"
        experiment = "SFCs:8-Topology:1"
        runExperiment(FGR8SFC, topology1)

    def experiment7() -> None:
        """
        Run Experiment 7.
        """

        # Experiment 7 - 32 SFCs 1
        nonlocal experiment
        global experimentName
        experimentName = "seven"
        experiment = "SFCs:32-Topology:1"
        runExperiment(FGR32SFC, topology1)

    def experiment8() -> None:
        """
        Run Experiment 8.
        """

        # Experiment 8 - 16 SFCs 1
        nonlocal experiment
        global experimentName

        experimentName = "eight"
        experiment = "SFCs:16-Topology:1"
        runExperiment(FGR16SFC, topology1)

    if experiment == 1:
        experiment1()
    elif experiment == 2:
        experiment2()
    elif experiment == 3:
        experiment3()
    elif experiment == 4:
        experiment4()
    elif experiment == 5:
        experiment5()
    elif experiment == 6:
        experiment6()
    elif experiment == 7:
        experiment7()
    elif experiment == 8:
        experiment8()
    else:
        experiment1()
        experiment2()
        experiment3()
        experiment4()
        experiment5()
        experiment6()
        experiment7()
        experiment8()
