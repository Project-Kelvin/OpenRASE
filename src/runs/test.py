"""
This file is used to test the functionality of the SFC Emulator.
"""

from time import sleep
import click
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.topology import SERVER, SFCC
from models.traffic_generator import TrafficData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.tui import TUI

topo: Topology = {
    "hosts": [
        {
            "id": "h1",
            "cpu": 4,
            "memory": 1024
        },
        {
            "id": "h2",
            "cpu": 4,
            "memory": 1024
        }
    ],
    "switches": [
        {
            "id": "s1"
        },
        {
            "id": "s2"
        }
    ],
    "links": [
        {
            "source": SFCC,
            "destination": "s1",
        },
        {
            "source": "s2",
            "destination": SERVER,
        },
        {
            "source": "h1",
            "destination": "s1",
            "bandwidth": 1000
        },
        {
            "source": "h2",
            "destination": "s2",
            "bandwidth": 1000
        },
        {
            "source": "s1",
            "destination": "s2",
            "bandwidth": 1000
        }
    ]
}

eg: EmbeddingGraph = {
    "sfcID": "sfc1",
    "vnfs": {
        "host": {
            "id": "h1"
        },
        "vnf": {
            "id": "waf"
        },
        "next": {
            "host": {
                "id": "h2"
            },
            "vnf": {
                "id": "ha"
            },
            "next": {
                "host": {
                    "id": SERVER
                },
                "next": TERMINAL
            }
        }
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
                "id": "h2"
            },
            "links": ["s1", "s2"]
        },
        {
            "source": {
                "id": "h2"
            },
            "destination": {
                "id": SERVER
            },
            "links": ["s2"]
        }
    ]
}

simpleEG: EmbeddingGraph = {
    "sfcID": "sfc2",
    "vnfs": {
        "host": {
            "id": "h1"
        },
        "vnf": {
            "id": "waf"
        },
        "next": {
            "host": {
                "id": SERVER
            },
            "next": TERMINAL
        }
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
            "links": ["s1", "s2"]
        }
    ]
}

simpleEGUpdated: EmbeddingGraph = {
    "sfcID": "sfc3",
    "vnfs": {
        "host": {
            "id": "h1"
        },
        "vnf": {
            "id": "lb"
        },
        "next": [{
            "host": {
                "id": SERVER
            },
            "next": TERMINAL
        }, {
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
            "links": ["s1", "s2"]
        }
    ]
}

sfcRequest: SFCRequest = {
    "sfcrID": "sfcr1",
    "latency": 100,
    "vnfs": ["lb", "ha", "tm", "waf"],
    "strictOrder": ["waf", "ha"],
}

trafficDesign: "list[TrafficDesign]" = [
    [
        {
            "target": 100,
            "duration": "30s"
        }
    ]
]

class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def generateRequests(self) -> None:

        self._orchestrator.sendRequests([sfcRequest])

class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def generateEmbeddingGraphs(self) -> None:
        """
        Generate the embedding graphs.
        """

        def updateTUI():
            TUI.appendToSolverLog("Starting traffic generation.")
            duration = 0.5*60
            elapsed = 0
            while elapsed < duration:
                sleep(5)
                data: "dict[str, TrafficData]" = self._trafficGenerator.getData("5s")

                for key, value in data.items():
                    httpReqs: int = value["httpReqs"]
                    averageLatency: float = value["averageLatency"]
                    httpReqsRate: float = httpReqs / 5 if httpReqs != 0 else 0
                    TUI.appendToSolverLog(f"{httpReqsRate} requests took {averageLatency} seconds on average for {key}.")
                    print(f"{httpReqsRate} requests took {averageLatency} seconds on average for {key}.")
                elapsed += 5
            TUI.appendToSolverLog("Solver has finished.")
        self._orchestrator.sendEmbeddingGraphs([simpleEG, simpleEGUpdated])
        updateTUI()
        self._orchestrator.sendEmbeddingGraphs([simpleEGUpdated])
        updateTUI()
        TUI.exit()


@click.command()
@click.option("--headless", default=False, type=bool, is_flag=True, help="Run the emulator in headless mode.")
def run (headless: bool) -> None:
    """
    Run the test.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.
    """

    sfcEmulator = SFCEmulator(SFCR, SFCSolver, headless)
    sfcEmulator.startTest(topo, trafficDesign)
    sfcEmulator.startCLI()
    sfcEmulator.end()
