"""
This file is used to test the functionality of the SFC Emulator.
"""

from time import sleep
from shared.models.forwarding_graph import ForwardingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from mano.infra_manager import InfraManager
from mano.vnf_manager import VNFManager
from constants.topology import SERVER, SFCC, TERMINAL

infraManager: InfraManager = InfraManager()
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

infraManager.installTopology(topo)
vnfManager: VNFManager = VNFManager(infraManager)
sleep(10)
vnfManager.deploySFF()
vnfManager.deployForwardingGraphs([fg])
