"""
This file is used to test the functionality of the SFC Emulator.
"""

from typing import Any
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from constants.notification import SFF_DEPLOYED
from constants.topology import SERVER, SFCC
from mano.infra_manager import InfraManager
from mano.notification_system import NotificationSystem, Subscriber
from mano.sdn_controller import SDNController
from mano.vnf_manager import VNFManager
from utils.container import getContainerIP


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

fg: EmbeddingGraph = {
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

simpleFG: EmbeddingGraph = {
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

sfcRequest: SFCRequest = {
    "sfcrID": "sfcr1",
    "latency": 100,
    "vnfs": ["lb, ha, tm, waf"],
    "strictOrder": ["waf, ha"],
}


class Test(Subscriber):
    """
    Test class.
    """

    def __init__(self):
        self._infraManager: InfraManager = InfraManager(SDNController())
        self._vnfManager: VNFManager = VNFManager(self._infraManager)
        NotificationSystem.subscribe(SFF_DEPLOYED, self)

    def startTest(self):
        """
        Start the test.
        """

        self._infraManager.installTopology(topo)

    def receiveNotification(self, topic: str, *args: "list[Any]"):
        if topic == SFF_DEPLOYED:
            self._vnfManager.deployEmbeddingGraphs([simpleFG])

    def getData(self):
        """
        Get the data.
        """

        print(self._infraManager.getTelemetry().getHostData())
        print(self._infraManager.getTelemetry().getSwitchData())

    def wait(self):
        """
        Wait for the test to finish.
        """

        self._infraManager.startCLI()
        self._infraManager.stopNetwork()

test = Test()
test.startTest()
test.getData()
print(getContainerIP(SFCC))
test.wait()
