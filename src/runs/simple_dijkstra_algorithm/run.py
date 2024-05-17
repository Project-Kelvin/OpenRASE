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
from packages.shared.models.sfc_request import SFCRequest
from shared.models.config import Config
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from calibrate.calibrate import Calibrate
from mano.orchestrator import Orchestrator
from models.calibrate import ResourceDemand
from packages.shared.models.embedding_graph import EmbeddingGraph
from packages.shared.models.topology import Topology
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from sfc.solver import Solver
from sfc.traffic_generator import TrafficGenerator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import calculateTrafficDuration, generateTrafficDesign
import click
from utils.tui import TUI

config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"

topology1: Topology = generateFatTreeTopology(4, 1000, 1, None)
topologyPointFive: Topology = generateFatTreeTopology(4, 1000, 0.5, None)
logFilePath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/experiments.csv"
hostDataFilePath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/host_data.csv"
latencyDataFilePath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/latency_data.csv"

with open(logFilePath, "w", encoding="utf8") as log:
    log.write("Experiment,Failed FGs,Accepted FGs,Execution Time,Deployment Time\n")

with open(hostDataFilePath, "w", encoding="utf8") as hostData:
    hostData.write("Experiment,Host,CPU,Memory,Duration\n")

with open(latencyDataFilePath, "w", encoding="utf8") as latencyData:
    latencyData.write("Experiment,SFC,Requests,Average Latency,Duration\n")

experiment: str = ""

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

        global experiment
        sfcEm: SFCEmulator = SFCEmulator(fgr, SFCSolver)
        sfcEm.startTest(topology, trafficDesign)
        sfcEm.end()
        experiment = ""

    def experiment1() -> None:
        """
        Run Experiment 1.
        """

        # Experiment 1 - 4 SFCs 0.5
        global experiment
        experiment = "SFCs:4-Topology:0.5"
        runExperiment(FGR4SFC, topologyPointFive)

    def experiment2() -> None:
        """
        Run Experiment 2.
        """

        # Experiment 2 - 8 SFCs 0.5
        global experiment
        experiment = "SFCs:8-Topology:0.5"
        runExperiment(FGR8SFC, topologyPointFive)

    def experiment3() -> None:
        """
        Run Experiment 3.
        """

        # Experiment 3 - 32 SFCs 0.5
        experiment = "SFCs:32-Topology:0.5"
        runExperiment(FGR32SFC, topologyPointFive)

    def experiment4() -> None:
        """
        Run Experiment 4.
        """

        # Experiment 4 - 16 SFCs 0.5
        global experiment
        experiment = "SFCs:16-Topology:0.5"
        runExperiment(FGR16SFC, topologyPointFive)

    def experiment5() -> None:
        """
        Run Experiment 5.
        """

        # Experiment 5 - 4 SFCs 1
        global experiment
        experiment = "SFCs:4-Topology:1"
        runExperiment(FGR4SFC, topology1)

    def experiment6() -> None:
        """
        Run Experiment 6.
        """

        # Experiment 6 - 8 SFCs 1
        global experiment
        experiment = "SFCs:8-Topology:1"
        runExperiment(FGR8SFC, topology1)

    def experiment7() -> None:
        """
        Run Experiment 7.
        """

        # Experiment 7 - 32 SFCs 1
        global experiment
        experiment = "SFCs:22-Topology:1"
        runExperiment(FGR32SFC, topology1)

    def experiment8() -> None:
        """
        Run Experiment 8.
        """

        # Experiment 8 - 16 SFCs 1
        global experiment
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

        with open(f"{configPath}/forwarding-graphs.json", "r", encoding="utf8") as fgFile:
            fgs: "list[EmbeddingGraph]" = json.load(fgFile)
            for fg in fgs:
                self._fgs.append(copy.deepcopy(fg))

class FGR4SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs once.
    """

    def generateRequests(self) -> None:
        for index, fg in enumerate(self._fgs):
            fg["sfcrID"] = f"sfc{index}"

        self._orchestrator.sendRequests(self._fgs)

class FGR8SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs twice.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            fg["sfcrID"] = f"sfc{index}-1"
            copiedFG: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG["sfcrID"] = f"sfc{index}-2"
            copiedFGs.append(copiedFG)

        self._fgs.extend(copiedFGs)

        self._orchestrator.sendRequests(self._fgs)

class FGR32SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs 8 times.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            fg["sfcrID"] = f"sfc{index}-1"
            copiedFG: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG1: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG2: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG3: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG4: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG5: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG6: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG7: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG1["sfcrID"] = f"sfc{index}-2"
            copiedFG2["sfcrID"] = f"sfc{index}-3"
            copiedFG3["sfcrID"] = f"sfc{index}-4"
            copiedFG4["sfcrID"] = f"sfc{index}-5"
            copiedFG5["sfcrID"] = f"sfc{index}-6"
            copiedFG6["sfcrID"] = f"sfc{index}-7"
            copiedFG7["sfcrID"] = f"sfc{index}-8"
            copiedFGs.append(copiedFG)
            copiedFGs.append(copiedFG1)
            copiedFGs.append(copiedFG2)
            copiedFGs.append(copiedFG3)
            copiedFGs.append(copiedFG4)
            copiedFGs.append(copiedFG5)
            copiedFGs.append(copiedFG6)
            copiedFGs.append(copiedFG7)

        self._fgs.extend(copiedFGs)

        self._orchestrator.sendRequests(self._fgs)

class FGR16SFC(FGR):
    """
    FG Request Generator that generates all 4 FGs four times.
    """

    def generateRequests(self) -> None:
        copiedFGs: "list[EmbeddingGraph]" = []
        for index, fg in enumerate(self._fgs):
            fg["sfcrID"] = f"sfc{index}-1"
            copiedFG: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG1: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG2: EmbeddingGraph = copy.deepcopy(fg)
            copiedFG["sfcrID"] = f"sfc{index}-2"
            copiedFG1["sfcrID"] = f"sfc{index}-3"
            copiedFG2["sfcrID"] = f"sfc{index}-4"
            copiedFGs.append(copiedFG)
            copiedFGs.append(copiedFG1)
            copiedFGs.append(copiedFG2)

        self._fgs.extend(copiedFGs)

        self._orchestrator.sendRequests(self._fgs)


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def __init__(self, orchestrator: Orchestrator, trafficGenerator: TrafficGenerator) -> None:
        super().__init__(orchestrator, trafficGenerator)
        self._resourceDemands: "dict[str, ResourceDemand]" = None

        calibrate = Calibrate()

        trafficDesignPath = f"{configPath}/traffic-design.json"
        with open(trafficDesignPath, "r", encoding="utf8") as traffic:
            design = json.load(traffic)
        maxTarget: int = max(design, key=lambda x: x["target"])["target"]

        self._resourceDemands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(maxTarget)

    def generateEmbeddingGraphs(self) -> None:
        try:
            while self._requests.empty():
                pass
            requests: "list[Union[FGR, SFCRequest]]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
            self._topology: Topology = self._orchestrator.getTopology()
            sda = SimpleDijkstraAlgorithm(requests, self._topology, self._resourceDemands)
            start: float = default_timer()
            fgs, failedFGs, _nodes = sda.run()
            end: float = default_timer()
            executionTime = end - start

            logRow: "list[str]" = []
            logRow.append(experiment)
            TUI.appendToSolverLog(f"Failed FGs: {len(failedFGs)}")
            TUI.appendToSolverLog(f"Accepted FGs: {len(fgs)}")
            logRow.append(str(len(failedFGs)))
            logRow.append(str(len(fgs)))
            TUI.appendToSolverLog(f"Acceptance Ratio: {len(fgs) / (len(fgs) + len(failedFGs)) * 100:.2f}%")
            logRow.append(f"{executionTime:.6f}")
            TUI.appendToSolverLog(f"Execution Time: {executionTime:.6f}s")

            TUI.appendToSolverLog(f"Deploying Embedding Graphs.")
            start = default_timer()
            self._orchestrator.sendEmbeddingGraphs(fgs)
            end = default_timer()
            TUI.appendToSolverLog(f"Finished deploying Embedding Graphs.")
            deploymentTime = end - start

            logRow.append(f"{deploymentTime:.6f}")
            TUI.appendToSolverLog(f"Deployment Time: {deploymentTime:.6f}s")

            with open(logFilePath, "a", encoding="utf8") as log:
                log.write(f"{','.join(logRow)}\n")

            trafficDuration: int = calculateTrafficDuration(self._trafficGenerator.getDesign()[0])
            TUI.appendToSolverLog(f"Waiting for {trafficDuration}s.")
            time: int = 0
            telemetry: Telemetry = self._orchestrator.getTelemetry()

            try:
                while time < trafficDuration:
                    start: float = default_timer()
                    hostData: HostData = telemetry.getHostData()
                    end: float = default_timer()
                    duration: int = round(end - start, 0)
                    trafficData: "dict[str, TrafficData]" = self._trafficGenerator.getData(
                        f"{duration:.0f}s")

                    for key, data in hostData.items():
                        hostRow: "list[str]" = []
                        hostRow.append(experiment)
                        hostRow.append(key)

                        hostRow.append(str(data["cpuUsage"][0]))
                        hostRow.append(str(data["memoryUsage"][0]/(1024*1024) if data["memoryUsage"][0] != 0 else 0))
                        hostRow.append(str(duration))
                        with open(hostDataFilePath, "a", encoding="utf8") as hostDataFile:
                            hostDataFile.write(f"{','.join(hostRow)}\n")

                    for key, data in trafficData.items():
                        row: "list[str]" = []
                        row.append(experiment)
                        row.append(key)
                        row.append(str(data["httpReqs"]))
                        row.append(str(data["averageLatency"]))
                        row.append(str(duration))
                        with open(latencyDataFilePath, "a", encoding="utf8") as latencyDataFile:
                            latencyDataFile.write(f"{','.join(row)}\n")

                    time += duration
            except Exception as e:
                TUI.appendToSolverLog("c"+str(e), True)

            TUI.appendToSolverLog(f"Finished waiting.")

            data: "dict[str, TrafficData]" = self._trafficGenerator.getData(f"{trafficDuration}s")

            TUI.appendToSolverLog(f"Finished experiment.")
            sleep(2)
        except Exception as e:
            TUI.appendToSolverLog(str(e), True)
        TUI.exit()

def getTrafficDesign() -> None:
    """
    Get the Traffic Design.
    """

    design: TrafficDesign = generateTrafficDesign(
        f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/data/requests.csv", 2)

    with open(f"{configPath}/traffic-design.json", "w", encoding="utf8") as traffic:
        json.dump(design, traffic, indent=4)
