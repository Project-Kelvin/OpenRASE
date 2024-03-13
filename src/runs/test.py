"""
This file is used to test the functionality of the SFC Emulator.
"""

from time import sleep
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.topology import SERVER, SFCC
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver

topo: Topology = {
    "hosts": [
        {
            "id": "h1",
            "cpu": 4,
            "memory": "1gb"
        },
        {
            "id": "h2",
            "cpu": 4,
            "memory": "1gb"
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
    "sfcID": "sfc2",
    "vnfs": {
        "host": {
            "id": "h2"
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
            "duration": "5m"
        }
    ]
]

class SFCR(SFCRequestGenerator):
    """
    SFC Request Generator.
    """

    def generateRequests(self) -> None:

        self._orchestrator.sendSFCRequests([sfcRequest])

class SFCSolver(Solver):
    """
    SFC Solver.
    """

    def generateEmbeddingGraphs(self) -> None:
        """
        Generate the embedding graphs.
        """

        self._orchestrator.sendEmbeddingGraphs([simpleEG])
        sleep(120)
        self._orchestrator.sendEmbeddingGraphs([simpleEGUpdated])

sfcEmulator = SFCEmulator(SFCR, SFCSolver)
sfcEmulator.startTest(topo, trafficDesign)
sfcEmulator.startCLI()
sfcEmulator.end()
