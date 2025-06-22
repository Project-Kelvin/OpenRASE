"""
This defines the GA that evolves teh weights of the Neural Network.
"""

import random
from multiprocessing import Pool, cpu_count
from typing import Any
from deap import base, creator, tools
import tensorflow as tf
from shared.utils.config import getConfig
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.hybrid_online_offline import decodePop
from algorithms.surrogacy.utils.extract_weights import getWeightLength
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from algorithms.surrogacy.utils.scorer import Scorer
from utils.tui import TUI


tf.get_logger().setLevel("ERROR")

artifactDir: str = f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy"
hybridEvolution: HybridEvolution = HybridEvolution()

def evaluate(
    decodedIndividual: DecodedIndividual,
    gen: int,
    ngen: int,
    trafficDesign: TrafficDesign,
    topology: Topology
) -> "tuple[int, float, float, float]":
    """
    Evaluates the individual.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        gen (int): the generation.
        ngen (int): the number of generations.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.

    Returns:
        tuple[int, float, float, float]: the individual index, acceptance ratio, max CPU usage, max memory usage.
    """

    penaltyScore: float = 50
    penalty: float = gen / ngen

    _index, egs, embedData, linkData, acceptanceRatio = decodedIndividual

    if len(egs) > 0:
        maxCPU, maxMemory = hybridEvolution.getMaxCpuMemoryUsageOfHosts(egs, topology, embedData, trafficDesign)

        return acceptanceRatio, maxCPU, maxMemory
    else:
        maxCPU: float = penalty * penaltyScore
        maxMemory: float = penalty * penaltyScore

        return acceptanceRatio, maxCPU, maxMemory


def evolveInitialWeights(
    popSize: int,
    fgs: "list[EmbeddingGraph]",
    trafficDesign: TrafficDesign,
    topology: Topology,
    minAR: float,
    maxCPU: float,
    maxMemory: float,
    dataType: int,
) -> "list[list[float]]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        popSize (int): the population size.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        minAR (float): The minimum acceptance ratio.
        maxCPU (float): The maximum CPU demand.
        maxMemory (float): The maximum memory demand.
        dataType (int): The data type (LC LL:0, LC HL:1, HC LL:2, HC HL:3 ).

    Returns:
        list[list[float]]: the weights.
    """

    TUI.appendToSolverLog("Evolving initial population.")

    MULTIPLIER: int = 10
    POP_SIZE: int = popSize * MULTIPLIER
    NGEN: int = 200
    CXPB: float = 1.0
    MUTPB: float = 1.0
    MIN_IND: int = round(popSize * 0.8)

    creator.create("MaxHosts", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxHosts)

    toolbox: base.Toolbox = base.Toolbox()
    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=getWeightLength(fgs, topology),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.9)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    decodedPop: "list[DecodedIndividual]" = decodePop(pop, topology, fgs)
    hybridEvolution.cacheForOnline(decodedPop, trafficDesign)
    pool: Any = Pool(processes=cpu_count())
    results: tuple[float, float, float] = pool.starmap(
        evaluate,
        [
            (
                ind,
                gen,
                NGEN,
                trafficDesign,
                topology,
            )
            for i, ind in enumerate(decodedPop)
        ],
    )

    for result in results:
        ind: "creator.Individual" = pop[result[0]]
        ind.fitness.values = (result[1], result[2])

    gen = gen + 1
    filteredPop: "list[creator.Individual]" = []
    while len(filteredPop) < MIN_IND:
        offspring: "list[creator.Individual]" = list(map(toolbox.clone, pop))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        decodedOffspring: "list[DecodedIndividual]" = decodePop(
            offspring, topology, fgs
        )
        hybridEvolution.cacheForOnline(decodedOffspring, trafficDesign)
        pool: Any = Pool(processes=cpu_count())
        results: tuple[float, float, float] = pool.starmap(
            evaluate,
            [
                (
                    ind,
                    gen,
                    NGEN,
                    trafficDesign,
                    topology,
                )
                for i, ind in enumerate(decodedOffspring)
            ],
        )

        for result in results:
            ind: "creator.Individual" = pop[result[0]]
            ind.fitness.values = (result[1], result[2])

        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)
        alpha: float = 1.0
        pop[:] = random.sample(pop, int(alpha * POP_SIZE)) + random.sample(
            offspring, int((1 - alpha) * POP_SIZE)
        )

        firstFilteredPop: "list[creator.Individual]" = [
            ind
            for ind in pop
            if ind.fitness.values[0] >= minAR
            and ind.fitness.values[2] < maxMemory
        ]

        if dataType == 0 or dataType == 1:
            filteredPop = [
                ind
                for ind in firstFilteredPop
                if ind.fitness.values[1] < maxCPU
            ]
        else:
            filteredPop = [
                ind
                for ind in firstFilteredPop
                if ind.fitness.values[1] > 1.1
                and ind.fitness.values[1] <= 2
            ]

        TUI.appendToSolverLog(f"Generation {gen} completed.")
        TUI.appendToSolverLog(
            f"{len(filteredPop)} individuals satisfy the criteria."
        )
        gen = gen + 1

    del creator.Individual

    doubledPop = filteredPop * 2
    newPop: "list[creator.Individual]" = []
    if len(doubledPop) < popSize:
        rem: int = popSize - len(doubledPop)
        newPop = doubledPop + random.sample(pop, rem)
    else:
        rem: int = popSize - len(filteredPop)
        newPop = filteredPop + random.sample(pop, rem)

    return newPop
