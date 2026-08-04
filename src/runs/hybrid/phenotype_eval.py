"""
This defines the function to evaluate GENESIS'S ability to explore the phenotype space.
"""

import os
from algorithms.ga_dijkstra_algorithm.bega_individual import convertIndividualToEmbeddingGraph as convertBEGAIndividualToEG, generateRandomIndividual as generateRandomBEGAIndividual
from algorithms.mak_ga.gaha_individual import convertIndividualToEmbeddingGraphs as convertGAHAIndividualToEG, generateRandomIndividual as generateRandomGAHAIndividual
from algorithms.utils.graphs import convertSFCRsToEGs
import click
from typing import Any, cast
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

def isAllCombinationsInDatabase(combinations: Any, database: dict[int, int]) -> bool:
    """
    Checks if all combinations are present in the database.

    Parameters:
        combinations (Any): A list of combinations to check.
        database (dict[int, int]): The database to check.

    Returns:
        bool: True if all combinations are present, False otherwise.
    """

    for c in range(len(combinations)):
        if not isCombinationInDatabase(c, database):
            return False

    return True

def calculateDiscoveredPercentage(disCoveredDatabase: dict[int, int], database: dict[int, int]) -> float:
    """
    Calculates the percentage of discovered combinations in the database.

    Parameters:
        disCoveredDatabase (dict[int, int]): The database of discovered combinations.
        database (dict[int, int]): The database to check.

    Returns:
        float: The percentage of discovered combinations.
    """

    return round((len(disCoveredDatabase) / len(database)) * 100, 2) if len(database) > 0 else 0.0

def getCombinationIndex(combination: Any, combinations: Any) -> int:
    """
    Retrieves the index of a given combination in the combinations list.

    Parameters:
        combination (Any): The combination to find.
        combinations (Any): The list of combinations to search.

    Returns:
        int: The index of the combination in the combinations list, or -1 if not found.
    """

    for i, c in enumerate(combinations):
        if c == combination:

            return i

    return -1

@click.command()
@click.option("--algo", type=click.Choice(["genesis", "gaha", "bega"]), default="genesis", help="Run the specified algorithm.")
@click.option("--dijkstra", is_flag=True, default=False, help="Run GENESIS with Dijkstra's algorithm.")
@click.option("--chain", is_flag=True, default=False, help="Run GENESIS with static chain-based evaluation.")
def run(algo: str, dijkstra: bool, chain: bool) -> None:
    """
    This function runs the specified algorithm on the phenotype space and evaluates its performance.

    Parameters:
        algo (str): The algorithm to run.
        dijkstra (bool): Flag to run GENESIS with Dijkstra's algorithm.
        chain (bool): Flag to run GENESIS with static chain-based evaluation.
        dijkstra (bool): Flag to run GENESIS with Dijkstra's algorithm.
        chain (bool): Flag to run GENESIS with static chain-based evaluation.

    """

    TUI.disable()

    topology: Topology = cast(Topology, {
        "hosts": [
            {
                "id": "h1"
            },
            {
                "id": "h2"
            },
            {
                "id": "h3"
            },
            {
                "id": "h4"
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
                "destination": "s2",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "h1",
                "destination": "s1",
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
                "source": "s1",
                "destination": "s3",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "h3",
                "destination": "s3",
                "bandwidth": 1000,
                "delay": 10
            },
            {
                "source": "h4",
                "destination": "s2",
                "bandwidth": 1000,
                "delay": 10
            }
        ]
    })

    sfcrs: list[SFCRequest] = [
        cast(SFCRequest, {
            "sfcrID": "sfcr1",
            "vnfs": ["waf", "tm", "ha", "ips"]
        })
    ]

    ccOrder: list[list[str]] = []
    for vnf1 in sfcrs[0]["vnfs"]:
        for vnf2 in sfcrs[0]["vnfs"]:
            if vnf2 == vnf1:
                continue
            for vnf3 in sfcrs[0]["vnfs"]:
                if vnf3 == vnf1 or vnf3 == vnf2:
                    continue
                for vnf4 in sfcrs[0]["vnfs"]:
                    if vnf4 == vnf1 or vnf4 == vnf2 or vnf4 == vnf3:
                        continue
                    ccOrder.append([vnf1, vnf2, vnf3, vnf4])

    hostOrder: list[list[str]] = []

    for host1 in topology["hosts"]:
        for host2 in topology["hosts"]:
            for host3 in topology["hosts"]:
                for host4 in topology["hosts"]:
                    hostOrder.append([host1["id"], host2["id"], host3["id"], host4["id"]])

    linksOrder: dict[str, list[list[str]]] = {
        "sfcc-h1": [["s1"]],
        "sfcc-h2": [["s1", "s2"], ["s1", "s3", "s2"]],
        "sfcc-h3": [["s1","s3"], ["s1", "s2", "s3"]],
        "sfcc-h4": [["s1", "s2"], ["s1", "s3", "s2"]],
        "h1-h1": [[]],
        "h2-h2": [[]],
        "h3-h3": [[]],
        "h4-h4": [[]],
        "h1-server": [["s1", "s2", "s3"], ["s1", "s3"]],
        "h2-server": [["s2", "s3"], ["s2", "s1", "s3"]],
        "h3-server": [["s3"]],
        "h4-server": [["s2", "s3"], ["s2", "s1", "s3"]],
        "h1-h3": [["s1", "s3"], ["s1", "s2", "s3"]],
        "h3-h1": [["s3", "s1"], ["s3", "s2", "s1"]],
        "h2-h3": [["s2", "s3"], ["s2", "s1", "s3"]],
        "h3-h2": [["s3", "s2"], ["s3", "s1", "s2"]],
        "h1-h2": [["s1", "s2"], ["s1", "s3", "s2"]],
        "h2-h1": [["s2", "s1"], ["s2", "s3", "s1"]],
        "h1-h4": [["s1", "s2"], ["s1", "s3", "s2"]],
        "h4-h1": [["s2", "s1"], ["s2", "s3", "s1"]],
        "h2-h4": [["s2"]],
        "h4-h2": [["s2"]],
        "h3-h4": [["s3", "s2"], ["s3", "s1", "s2"]],
        "h4-h3": [["s2", "s3"], ["s2", "s1", "s3"]]
    }

    combinations: list[tuple[list[str], list[str], tuple[list[str], list[str], list[str], list[str], list[str]]]] = []
    for cc in ccOrder:
        for host in hostOrder:
            for sfccLink in linksOrder[f"sfcc-{host[0]}"]:
                for hostLink in linksOrder[f"{host[0]}-{host[1]}"]:
                    for host1Link in linksOrder[f"{host[1]}-{host[2]}"]:
                        for host2Link in linksOrder[f"{host[2]}-{host[3]}"]:
                            for serverLink in linksOrder[f"{host[3]}-server"]:
                                combinations.append((cc, host, (sfccLink, hostLink, host1Link, host2Link, serverLink)))

    hostCCCombinations: list[tuple[list[str], list[str]]] = []
    for cc in ccOrder:
        for host in hostOrder:
            hostCCCombinations.append((cc, host))

    hostLinksCombinations: list[tuple[list[str], tuple[list[str], list[str], list[str], list[str], list[str]]]] = []
    for host in hostOrder:
        for sfccLink in linksOrder[f"sfcc-{host[0]}"]:
            for hostLink in linksOrder[f"{host[0]}-{host[1]}"]:
                for host1Link in linksOrder[f"{host[1]}-{host[2]}"]:
                    for host2Link in linksOrder[f"{host[2]}-{host[3]}"]:
                        for serverLink in linksOrder[f"{host[3]}-server"]:
                            hostLinksCombinations.append((host, (sfccLink, hostLink, host1Link, host2Link, serverLink)))

    print("Total combinations:", len(combinations))
    print("Total CC combinations:", len(ccOrder))
    print("Total Host combinations:", len(hostOrder))
    print("Total CC-Host combinations:", len(hostCCCombinations))
    print("Total Host-Links combinations:", len(hostLinksCombinations))
    # for combination in combinations:
    #     print(combination)

    database: dict[int, int] = {
        c: 0 for c in range(len(combinations))
    }
    ccDatabase: dict[int, int] = {
        c: 0 for c in range(len(ccOrder))
    }
    hostDatabase: dict[int, int] = {
        c: 0 for c in range(len(hostOrder))
    }
    hostCCDatabase: dict[int, int] = {
        c: 0 for c in range(len(hostCCCombinations))
    }
    hostLinksDatabase: dict[int, int] = {
        c: 0 for c in range(len(hostLinksCombinations))
    }
    failed: int = 0
    allFound: bool = isAllCombinationsInDatabase(combinations, database)
    allCCFound: bool = isAllCombinationsInDatabase(ccOrder, ccDatabase)
    allHostFound: bool = isAllCombinationsInDatabase(hostOrder, hostDatabase)
    allCCHostFound: bool = isAllCombinationsInDatabase(hostCCCombinations, hostCCDatabase)
    allHostLinksFound: bool = isAllCombinationsInDatabase(hostLinksCombinations, hostLinksDatabase)
    searched: int = 0
    population: list[Individual] = []
    discoveredCombinations: dict[int, int] = {}
    discoveredCCCombinations: dict[int, int] = {}
    discoveredHostCombinations: dict[int, int] = {}
    discoveredCCHostCombinations: dict[int, int] = {}

    while not allFound and searched < 100:
        eg: EmbeddingGraph = {}
        egs: list[EmbeddingGraph] = []
        if algo == "genesis":
            GenesisUtils.init(sfcrs, topology, 8, 0.00, 1.0, np.pi)
            individual: Individual = GenesisUtils.generateRandomGenesisIndividual(Individual, topology, sfcrs)
            population.append(individual)
            print("Random individual generated.")
            decodedIndividual: DecodedIndividual = GenesisUtils.decodeIndividual(cast(GenesisIndividual, individual), 0, topology, sfcrs, dijkstra=dijkstra, staticChain=chain)
            print("Individual decoded.")
            egs = decodedIndividual[1]
        elif algo == "gaha":
            individual: Individual = generateRandomGAHAIndividual(Individual, convertSFCRsToEGs(sfcrs), topology, 0.0)
            population.append(individual)
            print("Random individual generated.")
            egs, _, _, _, _= convertGAHAIndividualToEG(individual, topology, convertSFCRsToEGs(sfcrs), 0)
        elif algo == "bega":
            individual: Individual = generateRandomBEGAIndividual(Individual, topology, sfcrs, 0.0)
            population.append(individual)
            print("Random individual generated.")
            egs, _, _, _ = convertBEGAIndividualToEG(individual, sfcrs, topology, 0)

        if len(egs) == 0:
            print("Acceptance Rate is 0, cannot evaluate phenotype space.")
            failed += 1
            searched += 1

            continue

        eg = egs[0]

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

        egLinks: tuple[list[str], list[str],list[str], list[str], list[str]] = (
            egLinksOrder[f"sfcc-{egHostOrder[0]}"],
            egLinksOrder[f"{egHostOrder[0]}-{egHostOrder[1]}"] if f"{egHostOrder[0]}-{egHostOrder[1]}" in egLinksOrder else egLinksOrder[f"{egHostOrder[1]}-{egHostOrder[0]}"][::-1] if f"{egHostOrder[1]}-{egHostOrder[0]}" in egLinksOrder else [],
            egLinksOrder[f"{egHostOrder[1]}-{egHostOrder[2]}"] if f"{egHostOrder[1]}-{egHostOrder[2]}" in egLinksOrder else egLinksOrder[f"{egHostOrder[2]}-{egHostOrder[1]}"][::-1] if f"{egHostOrder[2]}-{egHostOrder[1]}" in egLinksOrder else [],
            egLinksOrder[f"{egHostOrder[2]}-{egHostOrder[3]}"] if f"{egHostOrder[2]}-{egHostOrder[3]}" in egLinksOrder else egLinksOrder[f"{egHostOrder[3]}-{egHostOrder[2]}"][::-1] if f"{egHostOrder[3]}-{egHostOrder[2]}" in egLinksOrder else [],
            egLinksOrder[f"{egHostOrder[3]}-server"]
        )

        generatedCombination: tuple[list[str], list[str], tuple[list[str], list[str], list[str], list[str], list[str]]] = (egCcOrder, egHostOrder, egLinks)

        print("Combination extracted.")

        dbCombinationIndex: int = getCombinationIndex(generatedCombination, combinations)
        ccDBCombinationIndex: int = getCombinationIndex(egCcOrder, ccOrder)
        hostDBCombinationIndex: int = getCombinationIndex(egHostOrder, hostOrder)
        hostCCCombinationsIndex: int = getCombinationIndex((egCcOrder, egHostOrder), hostCCCombinations)
        hostLinksCombinationsIndex: int = getCombinationIndex((egHostOrder, egLinks), hostLinksCombinations)
        database[dbCombinationIndex] += 1
        discoveredCombinations[dbCombinationIndex] = database[dbCombinationIndex]
        ccDatabase[ccDBCombinationIndex] += 1
        discoveredCCCombinations[ccDBCombinationIndex] = ccDatabase[ccDBCombinationIndex]
        hostDatabase[hostDBCombinationIndex] += 1
        discoveredHostCombinations[hostDBCombinationIndex] = hostDatabase[hostDBCombinationIndex]
        hostCCDatabase[hostCCCombinationsIndex] += 1
        discoveredCCHostCombinations[hostCCCombinationsIndex] = hostCCDatabase[hostCCCombinationsIndex]
        allCCFound: bool = isAllCombinationsInDatabase(ccOrder, ccDatabase)
        allHostFound: bool = isAllCombinationsInDatabase(hostOrder, hostDatabase)
        allFound = isAllCombinationsInDatabase(combinations, database)
        allCCHostFound: bool = isAllCombinationsInDatabase(hostCCCombinations, hostCCDatabase)
        allHostLinksFound: bool = isAllCombinationsInDatabase(hostLinksCombinations, hostLinksDatabase)
        searched += 1
        print("Searched:", searched, "\nFailed:", failed, "\nAll Found:", allFound, "\nAll CC Found:", allCCFound, "\nAll Host Found:", allHostFound, "\nAll CC-Host Found:", allCCHostFound, "\nAll Host-Links Found:", allHostLinksFound)

    print("Finished.")
    print("Writing overall results.")
    with open(os.path.join(phenotypeDir, "phenotype_eval.csv"), "w") as f:
        for combination, count in zip(combinations, database.values()):
            for vnf in combination[0]:
                f.write(f"{vnf},")
            for host in combination[1]:
                f.write(f"{host},")
            for link in combination[2]:
                f.write(f"{'->'.join(link)},")
            f.write(f"{count}\n")

    print("Writing CC results.")
    with open(os.path.join(phenotypeDir, "phenotype_cc_eval.csv"), "w") as f:
        for combination, count in zip(ccOrder, ccDatabase.values()):
            for vnf in combination:
                f.write(f"{vnf},")
            f.write(f"{count}\n")

    print("Writing Host results.")
    with open(os.path.join(phenotypeDir, "phenotype_host_eval.csv"), "w") as f:
        for combination, count in zip(hostOrder, hostDatabase.values()):
            for host in combination:
                f.write(f"{host},")
            f.write(f"{count}\n")

    print("Writing CC-Host results.")
    with open(os.path.join(phenotypeDir, "phenotype_cc_host_eval.csv"), "w") as f:
        for combination, count in zip(hostCCCombinations, hostCCDatabase.values()):
            for vnf in combination[0]:
                f.write(f"{vnf},")
            for host in combination[1]:
                f.write(f"{host},")
            f.write(f"{count}\n")

    print("Writing Host-Links results.")
    with open(os.path.join(phenotypeDir, "phenotype_host_links_eval.csv"), "w") as f:
        for combination, count in zip(hostLinksCombinations, hostLinksDatabase.values()):
            for host in combination[0]:
                f.write(f"{host},")
            for link in combination[1]:
                f.write(f"{'->'.join(link)},")
            f.write(f"{count}\n")

    print("Writing summary results.")
    with open(os.path.join(phenotypeDir, "phenotype_summary.txt"), "w") as f:
        f.write(f"Algorithm: {algo}\n")
        f.write(f"Is Dijkstra: {dijkstra}\n")
        f.write(f"Is Static Chain: {chain}\n")
        f.write(f"Total combinations: {len(combinations)}\n")
        f.write(f"Total CC combinations: {len(ccOrder)}\n")
        f.write(f"Total Host combinations: {len(hostOrder)}\n")
        f.write(f"Total CC-Host combinations: {len(hostCCCombinations)}\n")
        f.write(f"Total searched: {searched}\n")
        f.write(f"Total failed: {failed}\n")
        f.write(f"All Found: {allFound}\n")
        f.write(f"All CC Found: {allCCFound}\n")
        f.write(f"All Host Found: {allHostFound}\n")
        f.write(f"All CC-Host Found: {allCCHostFound}\n")
        f.write(f"All Host-Links Found: {allHostLinksFound}\n")
        f.write(f"Discovered Percentage: {calculateDiscoveredPercentage(discoveredCombinations, database)}%\n")
        f.write(f"Discovered CC Percentage: {calculateDiscoveredPercentage(discoveredCCCombinations, ccDatabase)}%\n")
        f.write(f"Discovered Host Percentage: {calculateDiscoveredPercentage(discoveredHostCombinations, hostDatabase)}%\n")
        f.write(f"Discovered CC-Host Percentage: {calculateDiscoveredPercentage(discoveredCCHostCombinations, hostCCDatabase)}%\n")
        f.write(f"Discovered Host-Links Percentage: {calculateDiscoveredPercentage(discoveredCCHostCombinations, hostLinksDatabase)}%\n")
