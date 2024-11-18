"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
from typing import Any, Tuple
from algorithms.surrogacy.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from algorithms.surrogacy.scorer import Scorer
from models.calibrate import ResourceDemand
from utils.tui import TUI
import tensorflow as tf
from shared.models.traffic_design import TrafficDesign
from multiprocessing import Pool, cpu_count

tf.get_logger().setLevel("ERROR")

scorer: Scorer = Scorer()


def evaluate(
    individual: "list[float]",
    fgs: "list[EmbeddingGraph]",
    gen: int,
    ngen: int,
    trafficDesign: TrafficDesign,
    topology: Topology,
) -> "tuple[float, float, float]":
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
        tuple[float, float, float]: the acceptance ratio, max CPU usage, max memory usage.
    """

    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(
        individual, copiedFGs, topology
    )
    df: pd.DataFrame = convertFGstoDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, _nodes, embedData = convertDFtoFGs(newDF, copiedFGs, topology)

    penaltyScore: float = 50
    acceptanceRatio: float = len(egs) / len(fgs)
    penalty: float = gen / ngen

    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]
    if len(egs) > 0:
        # Validate EGs
        data: "dict[str, dict[str, float]]" = {
            eg["sfcID"]: {"reqps": maxReqps} for eg in egs
        }

        scorer.cacheData(data, egs)
        scores: "dict[str, ResourceDemand]" = scorer.getHostScores(
            data, topology, embedData
        )
        maxCPU: float = max([score["cpu"] for score in scores.values()])
        maxMemory: float = max([score["memory"] for score in scores.values()])

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
) -> "list[list[float]]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        popSize (int): the population size.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        maxCPU (float): The maximum CPU demand.
        maxMemory (float): The maximum memory demand.
        minAR (float): The minimum acceptance ratio.

    Returns:
        list[list[float]]: the weights.
    """

    TUI.appendToSolverLog("Starting the evolution of the weights.")

    MULTIPLIER: int = 1
    POP_SIZE: int = popSize * MULTIPLIER
    NGEN: int = 200
    CXPB: float = 1.0
    MUTPB: float = 1.0
    maxCPU: float = 1
    maxMemory: float = 5
    minInd: int = POP_SIZE // 4

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
    pool: Any = Pool(processes=cpu_count())
    results: tuple[float, float, float] = pool.starmap(
        evaluate,
        [
            (
                ind,
                fgs,
                gen,
                NGEN,
                trafficDesign,
                topology,
            )
            for ind in pop
        ],
    )

    for ind, result in zip(pop, results):
        ind.fitness.values = result

    gen = gen + 1
    averageAR: float = 0
    averageCPU: float = 5
    ar1: "list[creator.Individual]" = []
    while len(ar1) < minInd:
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

        pool: Any = Pool(processes=cpu_count())
        results: tuple[float, float, float] = pool.starmap(
            evaluate,
            [
                (
                    ind,
                    fgs,
                    gen,
                    NGEN,
                    trafficDesign,
                    topology,
                )
                for ind in offspring
            ],
        )
        for ind, result in zip(offspring, results):
            ind.fitness.values = result

        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)
        alpha: float = 1.0
        pop[:] = random.sample(pop, int(alpha * POP_SIZE)) + random.sample(
            offspring, int((1 - alpha) * POP_SIZE)
        )

        ar1 = [
            ind
            for ind in pop
            if ind.fitness.values[0] >= minAR
            and ind.fitness.values[1] < maxCPU
            and ind.fitness.values[2] < maxMemory
        ]

        averageAR = np.mean([ind.fitness.values[0] for ind in pop])
        averageCPU = np.mean([ind.fitness.values[1] for ind in pop])

        TUI.appendToSolverLog(f"Generation {gen} completed.")
        TUI.appendToSolverLog(
            f"Average AR is {averageAR}. Max is {max([ind.fitness.values[0] for ind in pop])}. Min is {min([ind.fitness.values[0] for ind in pop])}."
        )
        TUI.appendToSolverLog(
            f"{len(ar1)} individuals have AR >= {minAR} and CPU <= {maxCPU}."
        )
        TUI.appendToSolverLog(
            f"Average max CPU is {averageCPU}. Max is {max([ind.fitness.values[1] for ind in pop])}. Min is {min([ind.fitness.values[1] for ind in pop])}."
        )
        TUI.appendToSolverLog(
            f"Average max memory is {np.mean([ind.fitness.values[2] for ind in pop])}. Max is {max([ind.fitness.values[2] for ind in pop])}. Min is {min([ind.fitness.values[2] for ind in pop])}."
        )
        gen = gen + 1

    del creator.Individual

    return pop
