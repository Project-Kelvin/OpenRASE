"""
This file is used to test the functionality of the SFC Emulator.
"""

from shared.constants.forwarding_graph import TERMINAL
from shared.models.forwarding_graph import ForwardingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from mano.infra_manager import InfraManager
from mano.sdn_controller import SDNController
from mano.vnf_manager import VNFManager
from constants.topology import SERVER, SFCC


infraManager: InfraManager = InfraManager(SDNController())
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

fg: ForwardingGraph = {
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
                "host":{
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

sfcRequest: SFCRequest = {
    "sfcrID": "sfcr1",
    "latency": 100,
    "vnfs": ["lb, ha, tm, waf"],
    "strictOrder": ["waf, ha"],
}

vnfManager: VNFManager = VNFManager(infraManager)
infraManager.installTopology(topo)
vnfManager.deployForwardingGraphs([fg])
print(infraManager.getTelemetry().getHostData())
print(infraManager.getTelemetry().getSwitchData())
infraManager.startCLI()
infraManager.stopNetwork()
