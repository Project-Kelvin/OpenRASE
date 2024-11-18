"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
from multiprocessing import Pool, cpu_count
import random
from time import sleep
from timeit import default_timer
from typing import Any, Callable, Tuple, Union
from algorithms.surrogacy.extract_weights import getWeightLength
from algorithms.surrogacy.generate import evolveInitialWeights, getWeights
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from algorithms.surrogacy.scorer import Scorer
from algorithms.surrogacy.surrogate import Surrogate
from models.calibrate import ResourceDemand
from shared.models.traffic_design import TrafficDesign
from sfc.traffic_generator import TrafficGenerator
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from utils.tui import TUI

scorer: Scorer = Scorer()


def evaluate(
    individual: "list[float]",
    fgs: "list[EmbeddingGraph]",
    gen: int,
    ngen: int,
    trafficDesign: TrafficDesign,
    topology: Topology,
    maxCPUDemand: float,
    maxMemoryDemand: float,
) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        gen (int): the generation.
        ngen (int): the number of generations.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        maxCPUDemand (float): maximum CPU demand.
        maxMemoryDemand (float): maximum memory demand.

    Returns:
        tuple[float, float]: the fitness.
    """

    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(
        individual, copiedFGs, topology
    )
    df: pd.DataFrame = convertFGstoDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, nodes, embedData = convertDFtoFGs(newDF, copiedFGs, topology)
    if len(egs) > 0:
        embedLinks: EmbedLinks = EmbedLinks(topology, egs, weights[2], weights[3])
        start: float = default_timer()
        egs = embedLinks.embedLinks(nodes)
        end: float = default_timer()
        TUI.appendToSolverLog(f"Link Embedding Time for all EGs: {end - start}s")

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs) / len(fgs)
    latency: int = 0
    penalty: float = gen / ngen
    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]
    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(fgs)} = {acceptanceRatio}"
    )
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
        TUI.appendToSolverLog(f"Max CPU: {maxCPU}, Max Memory: {maxMemory}")

        surrogate: Surrogate = Surrogate()

        reqs: "list[int]" = sum(
            [int(td["target"]) * int(td["duration"][:-1]) for td in trafficDesign[0]]
        )
        time: "list[int]" = sum([int(td["duration"][:-1]) for td in trafficDesign[0]])
        avg: float = reqs / time

        data: "dict[str, dict[str, float]]" = {
            eg["sfcID"]: {"reqps": avg, "latency": 0.0} for eg in egs
        }

        rows: "list[list[Union[str, float]]]" = scorer.getSFCScores(
            data, topology, egs, embedData, embedLinks.getLinkData()
        )

        for row in rows:
            row.append(len(egs))

        inputData: pd.DataFrame = pd.DataFrame(
            rows,
            columns=[
                "sfc",
                "reqps",
                "cpu",
                "avg_cpu",
                "memory",
                "avg_memory",
                "link",
                "latency",
                "sfc_hosts",
                "no_sfcs",
            ],
        )

        outputData: pd.DataFrame = surrogate.predict(inputData)

        latency = outputData["PredictedLatency"].mean()
        confidence: float = outputData["Confidence"].mean()

        TUI.appendToSolverLog(f"Latency: {latency}ms. Confidence: {confidence}")

        # Validate EGs
        # The resource demand of deployed VNFs exceeds the resource capacity of at least 1 host.
        # This leads to servers crashing.
        # Penalty is applied to the latency and the egs are not deployed.
        if maxCPU > maxCPUDemand or maxMemory > maxMemoryDemand:
            TUI.appendToSolverLog(
                f"Penalty because max CPU demand is {maxCPU} and max Memory demand is {maxMemory}."
            )
            latency = latency + penalty * (maxCPU + maxMemory)
            acceptanceRatio = acceptanceRatio - (penalty * (maxCPU + maxMemory))

            return acceptanceRatio, latency

    else:
        latency = penaltyLatency * penalty

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return acceptanceRatio, latency


def evolveUsingSurrogate(
    fgs: "list[EmbeddingGraph]",
    trafficDesign: TrafficDesign,
    topology: Topology,
    popSize: int,
    minAR: float,
    maxLatency: int,
) -> "list[creator.Individual]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        popSize (int): the population size.
        minAR (float): the minimum acceptance ratio.
        maxLatency (int): the maximum latency.

    Returns:
        None
    """

    TUI.appendToSolverLog(
        "Starting the evolution of the weights using the surrogate model."
    )

    POP_SIZE: int = 100
    NGEN: int = 5
    CXPB: float = 1.0
    MUTPB: float = 0.8
    maxCPUDemand: int = 1
    maxMemoryDemand: int = 5

    evolvedPop: "list[creator.Individual]" = evolveInitialWeights(
        POP_SIZE, fgs, trafficDesign, topology, minAR
    )
    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    evolvedNewPop: "list[creator.Individual]" = []
    for ep in evolvedPop:
        ind = creator.Individual()
        ind.extend(ep)
        evolvedNewPop.append(ind)

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
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.8)
    toolbox.register("select", tools.selNSGA2)

    randomPop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    alpha: float = 1.0
    pop: "list[creator.Individual]" = random.sample(
        evolvedNewPop, int(POP_SIZE * alpha)
    ) + random.sample(randomPop, int(POP_SIZE * (1 - alpha)))
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
                maxCPUDemand,
                maxMemoryDemand,
            )
            for ind in pop
        ],
    )

    for ind, result in zip(pop, results):
        ind.fitness.values = result

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    TUI.appendToSolverLog(
        f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
    )

    hof = tools.ParetoFront()
    hof.update(pop)

    selInds: "list[creator.Individual]" = []
    gen = gen + 1
    while len(selInds) < popSize:
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
                    maxCPUDemand,
                    maxMemoryDemand,
                )
                for ind in offspring
            ],
        )

        for ind, result in zip(offspring, results):
            ind.fitness.values = result
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        hof.update(pop)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        TUI.appendToSolverLog(
            f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )

        for ind in hof:
            TUI.appendToSolverLog(
                f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}"
            )

        gen = gen + 1

        selInds = [
            ind
            for ind in pop
            if ind.fitness.values[0] >= minAR and ind.fitness.values[1] <= maxLatency
        ]

    del creator.Individual
    del creator.MaxARMinLatency

    return selInds
