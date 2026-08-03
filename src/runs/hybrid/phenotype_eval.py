"""
This defines the function to evaluate GENESIS'S ability to explore the phenotype space.
"""

from typing import cast

import numpy as np
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology

from algorithms.hybrid.models.individuals import GenesisIndividual, Individual
from algorithms.hybrid.utils.extract_weights import generateRandomWeight
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.models.embedding import DecodedIndividual
from constants.topology import SERVER, SFCC
from utils.embedding_graph import traverseVNF


def run() -> None:
    """
    This function runs the GENESIS algorithm on the phenotype space and evaluates its performance.
    """

    topology: Topology = cast(Topology, {
        "hosts": [
            {
                "id": "h1"
            },
            {
                "id": "h2"
            }
        ],
        "switches": [
            {
                "id": "s1"
            },
            {
                "id": "s2"
            },
            {
                "id": "s3"
            },
            {
                "id": "s4"
            }
        ],
        "links": [
            {
                "source": SFCC,
                "destination": "s1",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "h2",
                "destination": "s4",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "h1",
                "destination": "s2",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": SERVER,
                "destination": "s3",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "s1",
                "destination": "s2",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "s2",
                "destination": "s3",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "s3",
                "destination": "s4",
                "bandwidth": 1000,
                "delay": 10
            }
        ]
    })

    sfcrs: list[SFCRequest] = [
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["waf", "tm"]
        })
    ]

    ccOrder: list[list[str]] = [["waf", "tm"], ["tm", "waf"]]
    hostOrder: list[list[str]] = [["h1", "h2"], ["h2", "h1"]]
    linksOrder: dict[str, list[list[str]]] = {
        "sfcc-h1": [["s1", "s2"], ["s1", "s4", "s3", "s2"]],
        "sfcc-h2": [["s1", "s4"], ["s1", "s2", "s3", "s4"]],
        "h1-h2": [["s2", "s3", "s4"], ["s2", "s1", "s4"]],
        "h2-h1": [["s4", "s3", "s2"], ["s4", "s1", "s2"]],
        "h1-server": [["s2", "s3"], ["s2", "s1", "s4", "s3"]],
        "h2-server": [["s4", "s3"], ["s4", "s1", "s2", "s3"]],
    }

    combinations: list[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]] = []

    for cc in ccOrder:
        for host in hostOrder:
            for sfccLink in linksOrder[f"sfcc-{host[0]}"]:
                for hostLink in linksOrder[f"{host[0]}-{host[1]}"]:
                    for serverLink in linksOrder[f"{host[1]}-server"]:
                        combinations.append((cc, host, (sfccLink, hostLink, serverLink)))

    GenesisUtils.init(sfcrs, topology, 2, 0.05, 1.0, np.pi)
    individual: Individual = GenesisUtils.generateRandomGenesisIndividual(Individual, topology, sfcrs)
    decodedIndividual: DecodedIndividual = GenesisUtils.decodeIndividual(cast(GenesisIndividual, individual), 0, topology, sfcrs)
    eg: EmbeddingGraph = decodedIndividual[1][0]

    egCcOrder: list[str] = []
    egHostOrder: list[str] = []
    egLinksOrder: dict[str, list[str]] = {}
    def parseVNF(vnf: VNF, _depth: int) -> None:
        """
        Parses a VNF and its children recursively, printing their details.

        Parameters:
            vnf (VNF): The VNF to parse.
            _depth (int): The current depth in the hierarchy for indentation.

        Returns:
            None
        """

        vnfId: str = vnf["vnfID"]
        egCcOrder.append(vnfId)
        hostId: str = vnf["hostID"]
        egHostOrder.append(hostId)

    traverseVNF(eg["vnfs"], parseVNF)

    for link in eg["links"]:
        egLinksOrder[f"{link['source']['id']}-{link['destination']['id']}"] = link["links"]

    egLinks: tuple[list[str], list[str], list[str]] = (
        egLinksOrder[f"sfcc-{egHostOrder[0]}"],
        egLinksOrder[f"{egHostOrder[0]}-{egHostOrder[1]}"],
        egLinksOrder[f"{egHostOrder[1]}-server"]
    )

    database: dict[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]], int] = {
        (combination[0], combination[1], combination[2]): 0 for combination in combinations
    }

    def isCombinationInDatabase(ccOrder: list[str], hostOrder: list[str], linksOrder: tuple[list[str], list[str], list[str]]) -> bool:
        """
        Checks if a given combination of ccOrder, hostOrder, and linksOrder exists in the database.

        Parameters:
            ccOrder (list[str]): The order of VNFs.
            hostOrder (list[str]): The order of hosts.
            linksOrder (tuple[list[str], list[str], list[str]]): The order of links.

        Returns:
            bool: True if the combination exists in the database, False otherwise.
        """

        for key in database.keys():
            if key[0] == ccOrder and key[1] == hostOrder and key[2] == (linksOrder[0], linksOrder[1], linksOrder[2]):
                return True
        return False

    def isAllCombinationsInDatabase() -> bool:
        """
        Checks if all combinations are present in the database.

        Returns:
            bool: True if all combinations are present, False otherwise.
        """

        for combination in combinations:
            if not isCombinationInDatabase(combination[0], combination[1], combination[2]):
                return False
        return True

    database[(egCcOrder, egHostOrder, egLinks)] += 1

    if isAllCombinationsInDatabase():
        print("All combinations are present in the database.")
