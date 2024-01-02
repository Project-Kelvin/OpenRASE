from shared.models.forwarding_graph import ForwardingGraph
from shared.models.topology import Topology
from mano.infra_manager import InfraManager
from constants.topology import TRAFFIC_GENERATOR, SFCC_SWITCH, SERVER, SFCC

infraManager = InfraManager()
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
            "id": "vnf1"
        },
        "next": {
            "host": {
                "id": "h2"
            },
            "vnf": {
                "id": "vnf2"
            },
            "next": {
                "host":{
                    "id": SERVER
                },
                "next": "terminal"
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

infraManager.installTopology(topo)
print(infraManager.assignIPs(fg))
