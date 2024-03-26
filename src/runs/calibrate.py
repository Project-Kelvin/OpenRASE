"""
This code is used to calibrate the CPU, memory, and bandwidth demands of VNFs.
"""

from timeit import default_timer
import csv
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.topology import SERVER, SFCC
from mano.telemetry import Telemetry
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver


sfcr: SFCRequest = {
    "sfcrID": "cWAF",
    "latency": 10000,
    "vnfs": ["waf"],
    "strictOrder": []
}

topology: Topology = {
    "hosts": [
        {
            "id": "h1",
        }
    ],
    "switches": [{"id": "s1"}],
    "links": [
        {
            "source": SFCC,
            "destination": "s1",
        },
        {
            "source": "s1",
            "destination": "h1"
        },
        {
            "source": "s1",
            "destination": SERVER
        }
    ]
}

eg: EmbeddingGraph = {
    "sfcID": "cWAF",
    "vnfs": {
        "host": {
            "id": "h1",
        },
        "vnf": {
            "id": "waf"
        },
        "next": [{
            "host": {
                "id": SERVER
            },
            "next": TERMINAL
        },
            {
            "host": {
                "id": SERVER
            },
            "next": TERMINAL
        }]
    },
    "links": [
        {
            "source": {
                "id": SFCC
            },
            "destination": {
                "id": "h1"
            },
            "links": ["s1"]
        },
        {
            "source": {
                "id": "h1"
            },
            "destination": {
                "id": SERVER
            },
            "links": ["s1"]
        }
    ]
}

trafficDesign: "list[TrafficDesign]" = [[
    {
        "duration": "5m",
        "target": 1000
    },
    {
        "duration": "5m",
        "target": 0
    },
    {
        "duration": "5m",
        "target": 2000
    },
    {
        "duration": "5m",
        "target": 0
    }
]]


class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def generateRequests(self) -> None:

        self._orchestrator.sendRequests([sfcr])


class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def generateEmbeddingGraphs(self) -> None:
        """
        Generate the embedding graphs.
        """

        self._orchestrator.sendEmbeddingGraphs([eg])
        telemetry: Telemetry = self._orchestrator.getTelemetry()
        seconds: int = 0

        headers = ["cpuUsage", "memoryUsage", "networkUsageIn",
                   "networkUsageOut", "http_reqs", "latency", "duration"]

        # Create a CSV file
        filename = "/home/thivi/SFC-Emulator/src/runs/calibrate.csv"

        # Write headers to the CSV file
        with open(filename, mode='w', newline='', encoding="utf8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
        while seconds < 20*60:
            start: float = default_timer()
            hostData: HostData = telemetry.getHostData()["h1"]["vnfs"]
            end: float = default_timer()
            duration: int = round(end - start, 0)
            trafficData: "list[TrafficData]" = self._trafficGenerator.getData(
                f"{duration:.0f}s")
            row = []

            for _key, data in hostData.items():
                row.append(data["cpuUsage"][0])
                row.append(data["memoryUsage"][0])
                row.append(data["networkUsage"][0])
                row.append(data["networkUsage"][1])

            for data in trafficData:
                row.append(data["value"])

            row.append(f"{round(end - start, 0):.0f}")

            # Write row data to the CSV file
            with open(filename, mode='a', newline='', encoding="utf8") as file:
                writer = csv.writer(file)
                writer.writerow(row)
            seconds += duration

        print("Finished!")


em: SFCEmulator = SFCEmulator(SFCR, SFCSolver)
em.startTest(topology, trafficDesign)
em.startCLI()
em.end()
