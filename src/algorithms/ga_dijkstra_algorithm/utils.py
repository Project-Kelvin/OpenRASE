"""
This defines a function that generates a random individual.
"""

from copy import deepcopy
import copy
import random
from time import sleep
from typing import Callable
from dijkstar import Graph, find_path
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from constants.topology import SERVER, SFCC
from deap import base
from models.calibrate import ResourceDemand
from models.traffic_generator import TrafficData
from packages.python.shared.models.embedding_graph import VNF, EmbeddingGraph
from packages.python.shared.models.topology import Topology
from packages.python.shared.models.traffic_design import TrafficDesign
from sfc.traffic_generator import TrafficGenerator
from utils.embedding_graph import traverseVNF
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI


def getVNFsfromFGRs(fgrs: "list[EmbeddingGraph]") -> "list[str]":
    """
    Get the VNFs from the SFC Request.

    Parameters:
        fgrs (list[EmbeddingGraph]): the FG Requests.

    Returns:
        list[str]: the VNFs.
    """

    vnfs: "list[str]" = []

    def parseVNF(vnf: VNF, _depth: int, vnfs: "list[str]") -> None:
        """
        Parse the VNF.

        Parameters:
            vnf (VNF): the VNF.
            _depth (int): the depth.

        Returns:
            None
        """

        vnfs.append(vnf["vnf"]["id"])

    for fgr in fgrs:
        traverseVNF(fgr["vnfs"], parseVNF, vnfs, shouldParseTerminal=False)

    return vnfs

def validateIndividual(individual: "list[list[int]]", topo: Topology, resourceDemands: "dict[str, ResourceDemand]", fgrs: "list[EmbeddingGraph]") -> bool:
    """
    Validate the individual.

    Parameters:
        individual (list[list[int]]): the individual to validate.
        topo (Topology): the topology.
        resourceDemands (dict[str, ResourceDemand]): the resource demands.
        fgrs (list[EmbeddingGraph]): the FG Requests.

    Returns:
        bool: True if the individual is valid, False otherwise.
    """

    vnfs: "list[str]" = getVNFsfromFGRs(fgrs)

    for index, host in enumerate(topo["hosts"]):
        totalDemand: ResourceDemand = ResourceDemand(cpu=0, memory=0, ior=0)
        for indexVNF, vnf in enumerate(vnfs):
            totalVNFs: int = 0
            if individual[indexVNF][index] == 1:
                totalVNFs += 1
                demand: ResourceDemand = resourceDemands[vnf]
                totalDemand["cpu"] += demand["cpu"]
                totalDemand["memory"] += demand["memory"]
                totalDemand["ior"] += demand["ior"]

        totalDemand["ior"] = totalDemand["ior"] / totalVNFs if totalVNFs > 0 else 0

        if totalDemand["cpu"] > host["cpu"]:
            return False

    return True

def generateRandomIndividual(container: list, topo: Topology, fgrs: "list[EmbeddingGraph]") -> "list[list[int]]":
    """
    Generate a random individual.

    Parameters:
        container (list): the container.
        topo (Topology): the topology.
        fgrs (EmbeddingGraph): the FG Request.

    Returns:
       list[list[int]]: the random individual.
    """

    individual: "list[list[int]]" = container()

    vnfs: "list[VNF]" = getVNFsfromFGRs(fgrs)
    noOfVNFs: int = len(vnfs)
    noOfHosts: int = len(topo["hosts"])

    for _ in range(noOfVNFs):
        item: "list[int]" = [0] * noOfHosts
        if random.random() < 0.999:
            item[random.randint(0, noOfHosts-1)] = 1
        individual.append(item)

    return individual

def convertIndividualToEmbeddingGraph(individual: "list[list[int]]", fgrs: "list[EmbeddingGraph]", topology: Topology)-> EmbeddingGraph:
    """
    Convert individual to an emebdding graph.

    Parameters:
        individual (list[list[int]]): the individual to convert.
        fgrs (list[EmbeddingGraph]): The SFC Requests.
        topology (Topology): The Topology.

    Returns:
        embedding graph (EmbeddingGraph): the embedding graph.
    """

    egs: "list[EmbeddingGraph]" = []
    offset: "list[int]" = [0]
    nodes: "dict[str, list[str]]" = {}

    for index, fgr in enumerate(fgrs):
        vnfs: VNF = fgr["vnfs"]
        embeddingNotFound: "list[bool]" = [False]
        fgr["sfcID"] = fgr["sfcrID"] if "sfcrID" in fgr else f"sfc{index}"
        nodes[fgr["sfcID"]] = [SFCC]

        def parseVNF(vnf: VNF, _depth: int, embeddingNotFound: "list[bool]", offset: "list[int]") -> None:
            """
            Parse the VNF.

            Parameters:
                vnf (VNF): the VNF.
                _depth (int): the depth.
                embeddingNotFound (list[bool]): the embedding not found.
                offset (list[int]): the offset.

            Returns:
                None
            """

            if embeddingNotFound[0]:
                return

            if "host" in vnf and vnf["host"]["id"] == SERVER:
                # pylint: disable=cell-var-from-loop
                nodes[fgr["sfcID"]].append(SERVER)

                return
            else:
                try:
                    vnf["host"] = {
                        "id": f"h{individual[offset[0]].index(1) + 1}"
                    }
                    # pylint: disable=cell-var-from-loop
                    if nodes[fgr["sfcID"]][-1] != vnf["host"]["id"]:
                        # pylint: disable=cell-var-from-loop
                        nodes[fgr["sfcID"]].append(vnf["host"]["id"])
                    offset[0] = offset[0] + 1
                except ValueError:
                    embeddingNotFound[0] = True

        traverseVNF(vnfs, parseVNF, embeddingNotFound, offset)

        if not embeddingNotFound[0]:
            if "sfcrID" in fgr:
                del fgr["sfcrID"]

            graph = Graph()
            nodePair: "list[str]" = []
            eg: EmbeddingGraph = copy.deepcopy(fgr)

            if "links" not in eg:
                eg["links"] = []

            for link in topology["links"]:
                graph.add_edge(
                    link["source"], link["destination"], link["bandwidth"])
                graph.add_edge(
                    link["destination"], link["source"], link["bandwidth"])

            for i in range(len(nodes[eg["sfcID"]]) - 1):
                srcDst: str = f"{nodes[eg['sfcID']][i]}-{nodes[eg['sfcID']][i + 1]}"
                dstSrc: str = f"{nodes[eg['sfcID']][i + 1]}-{nodes[eg['sfcID']][i]}"
                if srcDst not in nodePair and dstSrc not in nodePair:
                    nodePair.append(srcDst)
                    nodePair.append(dstSrc)
                    try:
                        path = find_path(graph, nodes[eg["sfcID"]][i], nodes[eg["sfcID"]][i + 1])
                    except Exception as e:
                        TUI.appendToSolverLog(f"Error: {e}")
                        continue

                    eg["links"].append({
                        "source": {"id": path.nodes[0]},
                        "destination": {"id": path.nodes[-1]},
                        "links": path.nodes[1:-1]
                    })

            egs.append(eg)

    return egs

def evaluation(individual: "list[list[int]]", fgrs: "list[EmbeddingGraph]", gen: int, ngen: int, sendEGs: "Callable[[list[EmbeddingGraph]], None]", deleteEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator, topology: Topology, resourceDemands: "dict[str, ResourceDemand]") -> "tuple[int]":
    """
    Evaluate the individual.

    Parameters:
        individual (list[list[int]]): the individual to evaluate.
        fgrs (list[EmbeddingGraph]): The SFC Requests.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): The Traffic Design.
        trafficGenerator (TrafficGenerator): The Traffic Generator.
        topology (Topology): The Topology.
        resourceDemands (dict(str, ResourceDemand)): The Resource Demands.

    Returns:
        tuple[int]: the evaluation.
    """

    if hasattr(individual.fitness, "values") and len(individual.fitness.values) > 0:
        TUI.appendToSolverLog(f"Individual already evaluated.")

        return individual.fitness.values

    egs: EmbeddingGraph = convertIndividualToEmbeddingGraph(individual, fgrs, topology)

    acceptanceRatio: float = len(egs) / len(fgrs)
    TUI.appendToSolverLog(f"Acceptance Ratio: {len(egs)}/{len(fgrs)} = {acceptanceRatio}")
    hosts = []
    penaltyLatency: int = 50000

    for vnf in individual:
        try:
            host = vnf.index(1)
            hosts.append(host)
        except ValueError:
            pass

    isValid: bool = validateIndividual(individual, topology, resourceDemands, fgrs)
    if isValid and len(egs) > 0:
        sendEGs(egs)

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")
        sleep(duration)
        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        trafficData: "dict[str, TrafficData]" = trafficGenerator.getData(
                        f"{duration:.0f}s")
        latency: float = 0
        for _key, value in trafficData.items():
            latency += value["averageLatency"]

        latency = latency / len(trafficData) if len(trafficData) > 0 else penaltyLatency

        TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
        deleteEGs(egs)
    else:
        penalty: float = gen/ngen
        acceptanceRatio = acceptanceRatio - penalty if len(egs) > 0 else acceptanceRatio
        latency = penaltyLatency * penalty if len(egs) > 0 else penaltyLatency

        if not isValid:
            TUI.appendToSolverLog(f"Invalid Individual.")

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return (acceptanceRatio, latency)


def mutate(individual: "list[list[int]]", indpb: float) -> "list[list[int]]":
    """
    Mutate the individual.

    Parameters:
        individual (list[list[int]]): the individual to mutate.
        indpb (float): the probability of mutation.

    Returns:
        list[list[int]]: the mutated individual.
    """

    mutatedIndividual: "list[list[int]]" = deepcopy(individual)

    for ind in mutatedIndividual:
        if random.random() < indpb:
            ind = [0] * len(ind)
            indices: "list[int]" = list(range(len(ind)))
            try:
                trueIndex: int = ind.index(1)
                indices.remove(trueIndex)
            except ValueError:
                pass
            ind[random.choice(indices)] = 1

    return mutatedIndividual

def algorithm(pop: "list[list[list[int]]]", toolbox: base.Toolbox, CXPB: float, MUTPB: float) -> "list[list[list[int]]]":
    """
    Run the algorithm.

    Parameters:
        pop (list[list[list[int]]]): the population.
        toolbox (base.Toolbox): the toolbox.
        CXPB (float): the crossover probability.
        MUTPB (float): the mutation probability.

    Returns:
        offspring (list[list[list[int]]]): the offspring.
    """

    offspring: "list[list[list[int]]]" = list(map(toolbox.clone, pop))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:

            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

            del mutant.fitness.values

    return offspring


def crossover(ind1: "list[list[int]]", ind2: "list[list[int]]") -> "tuple[list[list[int]], list[list[int]]]":
    """
    Crossover the individuals.

    Parameters:
        ind1 (list[list[int]]): the first individual.
        ind2 (list[list[int]]): the second individual.

    Returns:
        tuple[list[list[int]], list[list[int]]]: the crossovered individuals.
    """

    noOfVNFs: int = len(ind1)
    noOfHosts: int = len(ind1[0])

    xCutPoint: int = random.randint(1, noOfHosts-2)
    yCutPoint: int = random.randint(1, noOfVNFs-2)

    ind1Quads: "list[list[list[int]]]" = []
    ind2Quads: "list[list[list[int]]]" = []

    ind1ySlice1: "list[list[int]]" = ind1[:yCutPoint]
    ind1ySlice2: "list[list[int]]" = ind1[yCutPoint:]

    ind1xSlice1: "list[int]" = [vnf[:xCutPoint] for vnf in ind1ySlice1]
    ind1xSlice2: "list[int]" = [vnf[xCutPoint:] for vnf in ind1ySlice1]
    ind1xSlice3: "list[int]" = [vnf[:xCutPoint] for vnf in ind1ySlice2]
    ind1xSlice4: "list[int]" = [vnf[xCutPoint:] for vnf in ind1ySlice2]

    ind1Quads.append(ind1xSlice1)
    ind1Quads.append(ind1xSlice2)
    ind1Quads.append(ind1xSlice3)
    ind1Quads.append(ind1xSlice4)

    ind2ySlice1: "list[list[int]]" = ind2[:yCutPoint]
    ind2ySlice2: "list[list[int]]" = ind2[yCutPoint:]

    ind2xSlice1: "list[int]" = [vnf[:xCutPoint] for vnf in ind2ySlice1]
    ind2xSlice2: "list[int]" = [vnf[xCutPoint:] for vnf in ind2ySlice1]
    ind2xSlice3: "list[int]" = [vnf[:xCutPoint] for vnf in ind2ySlice2]
    ind2xSlice4: "list[int]" = [vnf[xCutPoint:] for vnf in ind2ySlice2]

    ind2Quads.append(ind2xSlice1)
    ind2Quads.append(ind2xSlice2)
    ind2Quads.append(ind2xSlice3)
    ind2Quads.append(ind2xSlice4)

    quads: "list[int]" = [0, 1, 2, 3]
    swapQ1: int = random.choice(quads)
    quads.remove(swapQ1)
    swapQ2: int = random.choice(quads)

    def fixMultiDeployment(ind1Q: "list[list[int]]", ind2Q: "list[list[int]]") -> "tuple[list[list[int]], list[list[int]]]":
        """
        Fix the multi deployment.

        Parameters:
            ind1 (list[list[int]]): the first individual.
            ind2 (list[list[int]]): the second individual.

        Returns:
            tuple[list[list[int]], list[list[int]]]: the fixed individuals.
        """

        for vnf1, vnf2 in zip(ind1Q, ind2Q):
            if not (vnf1.count(1) > 0 and vnf2.count(1)) > 0:
                continue
            fitness: int = random.randint(0, 1)
            if len(ind1.fitness.values) == 0:
                vnf2[vnf2.index(1)] = 0
            else:
                if fitness == 0:
                    if ind1.fitness.values[0] > ind2.fitness.values[0]:
                        vnf2[vnf2.index(1)] = 0
                    else:
                        vnf1[vnf1.index(1)] = 0
                else:
                    if ind1.fitness.values[1] < ind2.fitness.values[1]:
                        vnf2[vnf2.index(1)] = 0
                    else:
                        vnf1[vnf1.index(1)] = 0

    if swapQ1 % 2 == 0:
        fixMultiDeployment(ind1Quads[swapQ1], ind2Quads[swapQ1+1])
        fixMultiDeployment(ind1Quads[swapQ1+1], ind2Quads[swapQ1])
    else:
        fixMultiDeployment(ind1Quads[swapQ1], ind2Quads[swapQ1-1])
        fixMultiDeployment(ind1Quads[swapQ1-1], ind2Quads[swapQ1])

    if swapQ2 % 2 == 0:
        fixMultiDeployment(ind1Quads[swapQ2], ind2Quads[swapQ2+1])
        fixMultiDeployment(ind1Quads[swapQ2+1], ind2Quads[swapQ2])
    else:
        fixMultiDeployment(ind1Quads[swapQ2], ind2Quads[swapQ2-1])
        fixMultiDeployment(ind1Quads[swapQ2-1], ind2Quads[swapQ2])


    tempQ1 = ind1Quads[swapQ1]
    ind1Quads[swapQ1] = ind2Quads[swapQ1]
    ind2Quads[swapQ1] = tempQ1

    tempQ2 = ind1Quads[swapQ2]
    ind1Quads[swapQ2] = ind2Quads[swapQ2]
    ind2Quads[swapQ2] = tempQ2

    off1ySlice1: "list[list[int]]" = ind1Quads[0] + ind1Quads[2]
    off1ySlice2: "list[list[int]]" = ind1Quads[1] + ind1Quads[3]

    off1: "list[list[int]]" = [vnf1 + vnf2 for vnf1, vnf2 in zip(off1ySlice1, off1ySlice2)]

    off2ySlice1: "list[list[int]]" = ind2Quads[0] + ind2Quads[2]
    off2ySlice2: "list[list[int]]" = ind2Quads[1] + ind2Quads[3]

    off2: "list[list[int]]" = [vnf1 + vnf2 for vnf1, vnf2 in zip(off2ySlice1, off2ySlice2)]

    return off1, off2
