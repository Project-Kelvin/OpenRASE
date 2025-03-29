"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
from typing import Any, Tuple
from multiprocessing import Pool, cpu_count
import pandas as pd
from deap import base, creator, tools
import tensorflow as tf
from shared.utils.config import getConfig
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from algorithms.surrogacy.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.vnf_embedding import convertDFtoEGs, convertFGsToDF, getConfidenceValues
from algorithms.surrogacy.scorer import Scorer
from utils.tui import TUI
from models.calibrate import ResourceDemand


tf.get_logger().setLevel("ERROR")

scorer: Scorer = Scorer()

artifactDir: str = f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy"


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
    df: pd.DataFrame = convertFGsToDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, _nodes, embedData = convertDFtoEGs(newDF, copiedFGs, topology)

    penaltyScore: float = 50
    acceptanceRatio: float = len(egs) / len(fgs)
    penalty: float = gen / ngen

    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]
    if len(egs) > 0:
        # Validate EGs
        sfcIDs: "list[str]" = []
        reqps: "list[float]" = []
        for eg in egs:
            sfcIDs.append(eg["sfcID"])
            reqps.append(maxReqps)

        data: pd.DataFrame = pd.DataFrame(
            {
                "generation": 0,
                "individual": 0,
                "time": 0,
                "sfc": sfcIDs,
                "reqps": reqps,
                "real_reqps": 0,
                "latency": 0,
                "ar": acceptanceRatio,
            }
        )

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
