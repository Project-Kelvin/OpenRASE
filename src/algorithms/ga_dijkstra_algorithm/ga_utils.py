"""
This defines a function that generates a random individual.
"""

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import random
from time import sleep
import timeit
from typing import Callable, cast
from uuid import uuid4
import pandas as pd
from algorithms.ga_dijkstra_algorithm.bega_individual import convertIndividualToEmbeddingGraph
from shared.models.sfc_request import SFCRequest
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from algorithms.hybrid.utils.hybrid_evolution import Individual
from algorithms.models.embedding import DecodedIndividual
from calibrate.demand_predictor import DemandPredictor
from deap import base
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI


demandPredictor: DemandPredictor = DemandPredictor()

def evaluation(
    individual: DecodedIndividual,
    fgrs: "list[EmbeddingGraph]",
    gen: int,
    ngen: int,
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology
) -> "tuple[float, float]":
    """
    Evaluate the individual.

    Parameters:
        individual (DecodedIndividual): the individual to evaluate.
        fgrs (list[EmbeddingGraph]): The SFC Requests.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): The Traffic Design.
        trafficGenerator (TrafficGenerator): The Traffic Generator.
        topology (Topology): The Topology.
        maxTarget (int): The maximum target.

    Returns:
        tuple[float, float]: the evaluation.
    """

    egs: "list[EmbeddingGraph]" = individual[1]
    acceptanceRatio: float = individual[4]
    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(fgrs)} = {acceptanceRatio}"
    )
    penaltyLatency: int = 50000

    isValid: bool = not HybridEvaluation.doesExceedMemoryLimit(
        egs, topology, individual[2], trafficDesign, 2
    )
    if isValid and len(egs) > 0:
        sendEGs(egs)

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")
        sleep(duration)
        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        trafficData: pd.DataFrame = trafficGenerator.getData(f"{duration:.0f}s")

        latency: float = 0

        if (
            trafficData.empty
            or "_time" not in trafficData.columns
            or "_value" not in trafficData.columns
        ):
            TUI.appendToSolverLog("Traffic data is empty.")

            latency = penaltyLatency
        else:
            trafficData["_time"] = trafficData["_time"] // 1000000000

            groupedTrafficData: pd.DataFrame = trafficData.groupby(
                ["_time", "sfcID"]
            ).agg(
                reqps=("_value", "count"),
                medianLatency=("_value", "median"),
            )

            latency: float = groupedTrafficData["medianLatency"].mean()

        TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
        deleteEGs(egs)
    else:
        penalty: float = gen / ngen
        acceptanceRatio = acceptanceRatio - penalty if len(egs) > 0 else acceptanceRatio
        latency = penaltyLatency * penalty if len(egs) > 0 else penaltyLatency

        if not isValid:
            TUI.appendToSolverLog("Invalid Individual.")

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return (acceptanceRatio, round(latency))


def mutate(individual: Individual, indpb: float) -> Individual:
    """
    Mutate the individual.

    Parameters:
        individual (Individual): the individual to mutate.
        indpb (float): the probability of mutation.

    Returns:
        Individual: the mutated individual.
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


def algorithm(
    pop: "list[list[list[int]]]", toolbox: base.Toolbox, CXPB: float, MUTPB: float
) -> "list[list[list[int]]]":
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
    random.shuffle(offspring)
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:

            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values
            child1.id = uuid4()
            child2.id = uuid4()

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

            del mutant.fitness.values

    return offspring


def crossover(
    ind1: "list[list[int]]", ind2: "list[list[int]]"
) -> "tuple[list[list[int]], list[list[int]]]":
    """
    Crossover the individuals.

    Parameters:
        ind1 (list[list[int]]): the first individual.
        ind2 (list[list[int]]): the second individual.

    Returns:
        tuple[list[list[int]], list[list[int]]]: the crossbred individuals.
    """

    noOfVNFs: int = len(ind1)
    noOfHosts: int = len(ind1[0])

    xCutPoint: int = random.randint(1, noOfHosts - 2)
    yCutPoint: int = random.randint(1, noOfVNFs - 2)

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

    def fixMultiDeployment(
        ind1Q: "list[list[int]]", ind2Q: "list[list[int]]"
    ) -> "tuple[list[list[int]], list[list[int]]]":
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
        fixMultiDeployment(ind1Quads[swapQ1], ind2Quads[swapQ1 + 1])
        fixMultiDeployment(ind1Quads[swapQ1 + 1], ind2Quads[swapQ1])
    else:
        fixMultiDeployment(ind1Quads[swapQ1], ind2Quads[swapQ1 - 1])
        fixMultiDeployment(ind1Quads[swapQ1 - 1], ind2Quads[swapQ1])

    if swapQ2 % 2 == 0:
        fixMultiDeployment(ind1Quads[swapQ2], ind2Quads[swapQ2 + 1])
        fixMultiDeployment(ind1Quads[swapQ2 + 1], ind2Quads[swapQ2])
    else:
        fixMultiDeployment(ind1Quads[swapQ2], ind2Quads[swapQ2 - 1])
        fixMultiDeployment(ind1Quads[swapQ2 - 1], ind2Quads[swapQ2])

    tempQ1 = ind1Quads[swapQ1]
    ind1Quads[swapQ1] = ind2Quads[swapQ1]
    ind2Quads[swapQ1] = tempQ1

    tempQ2 = ind1Quads[swapQ2]
    ind1Quads[swapQ2] = ind2Quads[swapQ2]
    ind2Quads[swapQ2] = tempQ2

    off1ySlice1: "list[list[int]]" = ind1Quads[0] + ind1Quads[2]
    off1ySlice2: "list[list[int]]" = ind1Quads[1] + ind1Quads[3]

    off1: "list[list[int]]" = [
        vnf1 + vnf2 for vnf1, vnf2 in zip(off1ySlice1, off1ySlice2)
    ]

    off2ySlice1: "list[list[int]]" = ind2Quads[0] + ind2Quads[2]
    off2ySlice2: "list[list[int]]" = ind2Quads[1] + ind2Quads[3]

    off2: "list[list[int]]" = [
        vnf1 + vnf2 for vnf1, vnf2 in zip(off2ySlice1, off2ySlice2)
    ]

    return off1, off2


def decodePop(
    pop: "list[Individual]", topology: Topology, sfcrs: "list[SFCRequest]"
) -> "list[DecodedIndividual]":
    """
    Generate Embedding Graphs from the population.

    Parameters:
        pop (list[creator.Individual]): the population.
        topology (Topology): the topology.
        sfcrs (list[SFCRequest]): the SFC Requests.

    Returns:
        list[IndividualEG]: A list containing EGs, embedding data, link data and acceptance ratio.
    """

    startTime: float = timeit.default_timer()
    decodedPop: "list[DecodedIndividual]" = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                convertIndividualToEmbeddingGraph,
                individual,
                sfcrs,
                topology,
                index
            )
            for index, individual in enumerate(pop)
        ]

        for future in futures:
            egs, embeddingData, linkData, index = future.result()
            acceptanceRatio: float = len(egs) / len(sfcrs)
            decodedPop.append(cast(DecodedIndividual, (index, egs, embeddingData, linkData, acceptanceRatio, pop[index].id)))

    endTime: float = timeit.default_timer()
    TUI.appendToSolverLog(
        f"Decoded {len(decodedPop)} individuals in {endTime - startTime:.2f} seconds."
    )

    return decodedPop

def rejectVNF(individual: Individual, rejectionRate: float = 0.05) -> Individual:
    """
    Reject a VNF from the individual.

    Parameters:
        individual (Individual): the individual to reject VNF  from.
        rejectionRate (float): the probability of a VNF being deployed on a host.

    Returns:
        Individual: the individual with rejected VNF instances.
    """

    for ind in individual:
        if random.random() < rejectionRate:
            ind[ind.index(1)] = 0

    return individual
