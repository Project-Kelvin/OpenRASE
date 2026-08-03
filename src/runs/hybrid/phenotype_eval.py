"""
This defines the function to evaluate GENESIS'S ability to explore the phenotype space.
"""

import os
from typing import cast
import numpy as np
from packages.python.shared.constants.embedding_graph import TERMINAL
from packages.python.shared.utils.config import getConfig
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology

from algorithms.hybrid.models.individuals import GenesisIndividual, Individual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.models.embedding import DecodedIndividual
from constants.topology import SERVER, SFCC
from utils.embedding_graph import traverseVNF
from utils.tui import TUI

artifactsDir: str = os.path.join(getConfig()["repoAbsolutePath"], "artifacts")
if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)
experimentsDir: str = os.path.join(artifactsDir, "experiments")
if not os.path.exists(experimentsDir):
    os.makedirs(experimentsDir)
phenotypeDir: str = os.path.join(experimentsDir, "phenotype")
if not os.path.exists(phenotypeDir):
    os.makedirs(phenotypeDir)

def isCombinationInDatabase(c: int, database: dict[int, int]) -> bool:
    """
    Checks if a given combination of ccOrder, hostOrder, and linksOrder exists in the database.

    Parameters:
        c (int): The index of the combination to check.
        database (dict[int, int]): The database to check.

    Returns:
        bool: True if the combination exists in the database, False otherwise.
    """

    for key in database.keys():
        if key == c and database[key] > 0:
            return True

    return False

def isAllCombinationsInDatabase(combinations: list[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]], database: dict[int, int]) -> bool:
    """
    Checks if all combinations are present in the database.

    Parameters:
        combinations (list[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]]): A list of combinations to check.
        database (dict[int, int]): The database to check.

    Returns:
        bool: True if all combinations are present, False otherwise.
    """

    for c in range(len(combinations)):
        if not isCombinationInDatabase(c, database):
            return False
    return True

def getCombinationIndex(combination: tuple[list[str], list[str], tuple[list[str], list[str], list[str]]], combinations: list[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]]) -> int:
    """
    Retrieves the index of a given combination in the combinations list.

    Parameters:
        combination (tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]): The combination to find.
        combinations (list[tuple[list[str], list[str], tuple[list[str], list[str], list[str]]]]): The list of combinations to search.

    Returns:
        int: The index of the combination in the combinations list, or -1 if not found.
    """

    for i, c in enumerate(combinations):
        if c == combination:

            return i

    return -1

def run() -> None:
    """
    This function runs the GENESIS algorithm on the phenotype space and evaluates its performance.
    """

    TUI.disable()

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
    hostOrder: list[list[str]] = [["h1", "h2"], ["h2", "h1"], ["h1", "h1"], ["h2", "h2"]]
    linksOrder: dict[str, list[list[str]]] = {
        "sfcc-h1": [["s1", "s2"], ["s1", "s4", "s3", "s2"]],
        "sfcc-h2": [["s1", "s4"], ["s1", "s2", "s3", "s4"]],
        "h1-h2": [["s2", "s3", "s4"], ["s2", "s1", "s4"]],
        "h2-h1": [["s4", "s3", "s2"], ["s4", "s1", "s2"]],
        "h1-h1": [[]],
        "h2-h2": [[]],
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

    print("Total combinations:", len(combinations))

    database: dict[int, int] = {
        c: 0 for c in range(len(combinations))
    }
    failed: int = 0
    allFound: bool = isAllCombinationsInDatabase(combinations, database)
    searched: int = 0
    population: list[Individual] = []
    while not allFound and searched < 1000:
        GenesisUtils.init(sfcrs, topology, 8, 0.00, 1.0, np.pi)
        individual: Individual = GenesisUtils.generateRandomGenesisIndividual(Individual, topology, sfcrs)
        population.append(individual)
        print("Random individual generated.")
        decodedIndividual: DecodedIndividual = GenesisUtils.decodeIndividual(cast(GenesisIndividual, individual), 0, topology, sfcrs)
        print("Individual decoded.")

        if decodedIndividual[4] == 0:
            print("Acceptance Rate is 0, cannot evaluate phenotype space.")
            failed += 1

            continue

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

            if vnf["next"] == TERMINAL:
                return

            vnfId: str = vnf["vnf"]["id"]
            egCcOrder.append(vnfId)
            hostId: str = vnf["host"]["id"]
            egHostOrder.append(hostId)

        traverseVNF(eg["vnfs"], parseVNF)

        for link in eg["links"]:
            egLinksOrder[f"{link['source']['id']}-{link['destination']['id']}"] = link["links"]

        egLinks: tuple[list[str], list[str], list[str]] = (
            egLinksOrder[f"sfcc-{egHostOrder[0]}"],
            egLinksOrder[f"{egHostOrder[0]}-{egHostOrder[1]}"] if f"{egHostOrder[0]}-{egHostOrder[1]}" in egLinksOrder else [],
            egLinksOrder[f"{egHostOrder[1]}-server"]
        )

        generatedCombination: tuple[list[str], list[str], tuple[list[str], list[str], list[str]]] = (egCcOrder, egHostOrder, egLinks)

        print("Combination extracted.")

        database[getCombinationIndex(generatedCombination, combinations)] += 1
        allFound = isAllCombinationsInDatabase(combinations, database)
        searched += 1
        print("Searched:", searched, "Failed:", failed, "All Found:", allFound)

    with open(os.path.join(phenotypeDir, "phenotype_eval.csv"), "w") as f:
        for combination, count in zip(combinations, database.values()):
            for vnf in combination[0]:
                f.write(f"{vnf},")
            for host in combination[1]:
                f.write(f"{host},")
            for link in combination[2]:
                f.write(f"{'->'.join(link)},")
            f.write(f"{count}\n")
