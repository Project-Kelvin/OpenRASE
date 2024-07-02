"""
This defines a function that generates a random individual.
"""

from copy import deepcopy
import random
from time import sleep
from deap import base
from mano.vnf_manager import VNFManager
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

def generateRandomIndividual(noOfHosts: int, topo: Topology, resourceDemands: "dict[str, ResourceDemand]", fgrs: "list[EmbeddingGraph]") -> "list[list[int]]":
    """
    Generate a random individual.

    Parameters:
        noOfHosts (int): the number of hosts.
        topo (Topology): the topology.
        resourceDemands (dict[str, ResourceDemand]): the resource demands.
        fgr (EmbeddingGraph): the SFC Request.

    Returns:
       list[list[int]]: the random individual.
    """

    individual: "list[list[int]]" = []

    vnfs: "list[VNF]" = getVNFsfromFGRs(fgrs)
    noOfVNFs: int = len(vnfs)

    validIndividual: bool = False

    while not validIndividual:
        validIndividual = True
        for _ in range(noOfVNFs):
            item: "list[int]" = [0] * noOfHosts
            item[random.randint(0, noOfHosts-1)] = 1
            individual.append(item)

        validIndividual = validateIndividual(individual, topo, resourceDemands, fgrs)

    return individual

def convertIndividualToEmbeddingGraph(individual: "list[list[int]]", fgrs: "list[EmbeddingGraph]")-> EmbeddingGraph:
    """
    Convert individual to an emebdding graph.

    Parameters:
        individual (list[list[int]]): the individual to convert.
        fgrs (list[EmbeddingGraph]): The SFC Requests

    Returns:
        embedding graph (EmbeddingGraph): the embedding graph.
    """

    egs: "list[EmbeddingGraph]" = []
    offset: "tuple[int]" = (0,)

    for index, fgr in enumerate(fgrs):
        vnfs: VNF = fgr["vnfs"]
        embeddingNotFound: "tuple[bool]" = (False, )

        def parseVNF(vnf: VNF, _depth: int, embeddingNotFound: "tuple[bool]", offset: "tuple[int]") -> None:
            """
            Parse the VNF.

            Parameters:
                vnf (VNF): the VNF.
                _depth (int): the depth.
                embeddingNotFound (tuple[bool]): the embedding not found.
                offset (tuple[int]): the offset.

            Returns:
                None
            """

            if embeddingNotFound[0]:
                return

            try:
                vnf["host"] = {
                    "id": f"h{individual[offset[0]].index(1) + 1}"
                }
                offset = (offset[0] + 1,)
            except ValueError:
                embeddingNotFound = (True,)


        traverseVNF(vnfs, parseVNF, embeddingNotFound, offset, shouldParseTerminal=False)

        if not embeddingNotFound[0]:
            fgr["sfcID"] = fgr["sfcrID"] if "sfcrID" in fgr else f"sfc{index}"
            if "sfcrID" in fgr:
                del fgr["sfcrID"]

            egs.append(fgr)

    return egs

def evalutaionThunk(fgrs: "list[EmbeddingGraph]", vnfManager: VNFManager, trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator):
    """
    Return the evaluation function.

    Parameters:
        fgrs (list[EmbeddingGraph]): The SFC Requests
        vnfManager (VNFManager): The VNF Manager
        trafficDesign (TrafficDesign): The Traffic Design
        trafficGenerator (TrafficGenerator): The Traffic Generator

    Returns:
        function: the evaluation function.
    """

    def evaluation(individual: "list[list[int]]") -> "tuple[int]":
        """
        Evaluate the individual.

        Parameters:
            individual (list[list[int]]): the individual to evaluate.

        Returns:
            tuple[int]: the evaluation.
        """

        egs: EmbeddingGraph = convertIndividualToEmbeddingGraph(individual, fgrs)

        acceptanceRatio: float = len(egs) / len(fgrs)

        vnfManager.deployEmbeddingGraphs(egs)

        duration: int = calculateTrafficDuration(trafficDesign)
        time: int = 0

        while time < duration:
            waitDuration: int = 2
            sleep(waitDuration)
            TUI.appendToSolverLog(f"{duration-time}s more to go.")

        trafficData: "dict[str, TrafficData]" = trafficGenerator.getData(
                        f"{duration:.0f}s")
        latency: float = 0
        for _key, value in trafficData.items():
            latency = value["averageLatency"]

        latency = latency / len(trafficData)

        vnfManager.deleteEmbeddingGraphs(egs)

        return (acceptanceRatio, latency)

    return evaluation

def mutate(individual: "list[int]", indpb: float) -> "list[int]":
    """
    Mutate the individual.

    Parameters:
        individual (list[int]): the individual to mutate.
        indpb (float): the probability of mutation.

    Returns:
        list[int]: the mutated individual.
    """

    mutatedIndividual: "list[int]" = deepcopy(individual)

    for ind in mutatedIndividual:
        if random.random() < indpb:
            trueIndex: int = ind.index(1)
            ind = [0] * len(ind)
            indices: "list[int]" = list(range(len(ind)))
            indices.remove(trueIndex)
            ind[random.choice(indices)] = 1

    return mutatedIndividual

def algorithm(pop: "list[list[list[int]]]", toolbox: base.Toolbox, CXPB: float, MUTPB: float, topo: Topology, resourceDemands: "dict[str, ResourceDemand]", fgrs: "list[EmbeddingGraph]") -> "list[list[list[int]]]":
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
            validOffspring: bool = False
            child1Copy: "list[list[int]]" = []
            child2Copy: "list[list[int]]" = []

            while not validOffspring:
                validOffspring = True
                child1Copy = toolbox.clone(child1)
                child2Copy = toolbox.clone(child2)

                toolbox.mate(child1Copy, child2Copy)

                if not validateIndividual(child1Copy, topo, resourceDemands, fgrs) or not validateIndividual(child2Copy, topo, resourceDemands, fgrs):
                    validOffspring = False

            child1 = child1Copy
            child2 = child2Copy

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            mutantCopy: "list[list[int]]" = []
            validMutant: bool = False

            while not validMutant:
                validMutant = True
                mutantCopy = toolbox.clone(mutant)
                toolbox.mutate(mutantCopy)

                if not validateIndividual(mutantCopy, topo, resourceDemands, fgrs):
                    validMutant = False
            mutant = mutantCopy
            del mutant.fitness.values

    return offspring
