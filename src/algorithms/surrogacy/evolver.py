"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from multiprocessing import Pool, cpu_count
import random
from time import sleep
from typing import Any, Callable, Tuple
import os
import pandas as pd
import numpy as np
from deap import base, creator, tools
import tensorflow as tf
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig
from algorithms.surrogacy.constants.surrogate import (
    SURROGACY_PATH,
    SURROGATE_PATH,
)
from algorithms.surrogacy.utils.extract_weights import getWeightLength
from algorithms.surrogacy.utils.solvers import decodePop
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration, getTrafficDesignRate
from utils.tui import TUI

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

directory: str = SURROGACY_PATH
if not os.path.exists(directory):
    os.makedirs(directory)

surrogateDirectory: str = SURROGATE_PATH
if not os.path.exists(surrogateDirectory):
    os.makedirs(surrogateDirectory)

with open(
    f"{SURROGACY_PATH}/data.csv",
    "w",
    encoding="utf8",
) as topologyFile:
    topologyFile.write(
        "generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
    )

with open(
    f"{SURROGACY_PATH}/pfs.csv",
    "w",
    encoding="utf8",
) as pf:
    pf.write("generation, latency, ar\n")

isFirstSetWritten: bool = False
hybridEvolution: "HybridEvolution" = HybridEvolution()

def crossover(
    toolbox: base.Toolbox, pop: "list[creator.Individual]", cxpb: float, mutpb: float
) -> "list[creator.Individual]":
    """
    Crossover function.

    Parameters:
        toolbox (base.Toolbox): the toolbox.
        pop (list[creator.Individual]): the population.
        cxpb (float): the crossover probability.
        mutpb (float): the mutation probability.

    Returns:
        list[creator.Individual]: the offspring.
    """

    offspring: "list[creator.Individual]" = list(map(toolbox.clone, pop))
    random.shuffle(offspring)
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    return offspring


def select(
    offspring: "list[creator.Individual]",
    pop: "list[creator.Individual]",
    toolbox: base.Toolbox,
    popSize: int,
    hof: tools.ParetoFront,
) -> "Tuple[list[creator.Individual], tools.ParetoFront]":
    """
    Selection function.

    Parameters:
        offspring (list[creator.Individual]): the offspring.
        pop (list[creator.Individual]): the population.
        toolbox (base.Toolbox): the toolbox.
        popSize (int): the population size.
        hof (tools.ParetoFront): the hall of fame.

    Returns:
        Tuple[list[creator.Individual], tools.ParetoFront]: the population and the hall of fame.
    """

    pop[:] = toolbox.select(pop + offspring, k=popSize)

    hof.update(pop)

    return pop, hof


def writeData(gen: int, ars: "list[float]", latencies: "list[float]") -> None:
    """
    Writes the data to the file.

    Parameters:
        gen (int): the generation.
        ars (list[float]): the acceptance ratios.
        latencies (list[float]): the latencies.

    Returns:
        None
    """

    with open(
        f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv",
        "a",
        encoding="utf8",
    ) as dataFile:
        dataFile.write(
            f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )


def writePFs(gen: int, hof: tools.ParetoFront) -> None:
    """
    Writes the Pareto Fronts to the file.

    Parameters:
        gen (int): the generation.
        hof (tools.ParetoFront): the hall of fame.

    Returns:
        None
    """

    TUI.appendToSolverLog(f"Writing Pareto Fronts for generation {gen}.")
    for ind in hof:
        TUI.appendToSolverLog(
            f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}"
        )
        with open(
            f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv",
            "a",
            encoding="utf8",
        ) as pfFile:
            pfFile.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")


def evolveWeights(
    sfcrs: "list[SFCRequest]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Service Function Chains.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.

    Returns:
        None
    """

    POP_SIZE: int = 50
    NGEN: int = 100
    CXPB: float = 1.0
    MUTPB: float = 0.8
    MAX_MEMORY_DEMAND: int = 2
    MIN_QUAL_IND: int = 4
    MIN_AR: float = 0.5
    MAX_LATENCY: float = 500

    TUI.appendToSolverLog("Starting the evolution of the weights.")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=getWeightLength(sfcrs, topology),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.8)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1

    decodedPop: "list[creator.Individual]" = decodePop(pop, topology, sfcrs)
    hybridEvolution.cacheForOffline(
        decodedPop, trafficDesign, topology, gen, isAvgOnly=True
    )
    pool: Any = Pool(processes=cpu_count())
    results: tuple[float, float, float] = pool.starmap(
        hybridEvolution.evaluationOnSurrogate,
        [
            (ind, gen, NGEN, topology, trafficDesign, MAX_MEMORY_DEMAND)
            for ind in decodedPop
        ],
    )

    for result in results:
        ind: "creator.Individual" = pop[result[0]]
        ind.fitness.values = (result[1], result[2])

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies)

    hof = tools.ParetoFront()
    hof.update(pop)

    writePFs(gen, hof)

    gen = gen + 1

    qualifiedIndividuals: "list[creator.Individual]" = [
        ind
        for ind in hof
        if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
    ]

    while len(qualifiedIndividuals) < MIN_QUAL_IND:
        offspring: "list[creator.Individual]" = crossover(toolbox, pop, CXPB, MUTPB)
        decodedOffspring: "list[creator.Individual]" = decodePop(
            offspring, topology, sfcrs
        )
        hybridEvolution.cacheForOffline(
            decodedOffspring, trafficDesign, topology, gen, isAvgOnly=True
        )
        results: tuple[float, float, float] = pool.starmap(
            hybridEvolution.evaluationOnSurrogate,
            [
                (ind, gen, NGEN, topology, trafficDesign, MAX_MEMORY_DEMAND)
                for ind in decodedOffspring
            ],
        )

        for result in results:
            ind: "creator.Individual" = pop[result[0]]
            ind.fitness.values = (result[1], result[2])

        pop, hof = select(offspring, pop, toolbox, POP_SIZE, hof)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies)
        writePFs(gen, hof)

        qualifiedIndividuals = [
            ind
            for ind in hof
            if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
        ]

        minAR = min(ars)
        maxLatency = max(latencies)

        TUI.appendToSolverLog(
            f"Generation {gen}: Min AR: {minAR}, Max Latency: {maxLatency}"
        )

        TUI.appendToSolverLog(
            f"Qualified Individuals: {len(qualifiedIndividuals)}/{MIN_QUAL_IND}"
        )

        gen = gen + 1

    TUI.appendToSolverLog(
        f"Finished the evolution of weights using surrogate at generation {gen - 1}."
    )
    TUI.appendToSolverLog(
        f"Number of qualified individuals: {len(qualifiedIndividuals)}"
    )


    # ---------------------------------------------------------------------------------------------
    # Finished evolving the weights using surrogate.
    # Evolve using the emulator now.
    # ---------------------------------------------------------------------------------------------

    pop = qualifiedIndividuals
    emHof = tools.ParetoFront()
    emPopSize = len(pop)
    emQualifiedIndividuals: "list[creator.Individual]" = []
    EM_MIN_QUAL_IND: int = 1

    decodedPop: "list[creator.Individual]" = decodePop(pop, topology, sfcrs)
    hybridEvolution.cacheForOnline(
        decodedPop, trafficDesign
    )
    for ind in decodedPop:
        ind.fitness.values = hybridEvolution.evaluationOnEmulator(
            ind,
            sfcrs,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
            MAX_MEMORY_DEMAND,
        )

    emHof.update(pop)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies)
    writePFs(gen, emHof)

    emQualifiedIndividuals = [
        ind
        for ind in emHof
        if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
    ]

    emMinAR = min(ars)
    emMaxLatency = max(latencies)

    TUI.appendToSolverLog(
        f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
    )

    gen = gen + 1

    while len(emQualifiedIndividuals) < EM_MIN_QUAL_IND:
        offspring: "list[creator.Individual]" = []
        if len(pop) == 1:
            mutant: "creator.Individual" = toolbox.mutate(pop[0])
            del mutant.fitness.values
            offspring = [
                mutant
            ]
        else:
            offspring = crossover(toolbox, pop, CXPB, MUTPB)

        decodedOffspring: "list[creator.Individual]" = decodePop(
            offspring, topology, sfcrs
        )
        hybridEvolution.cacheForOnline(
            decodedOffspring, trafficDesign
        )
        for ind in decodedOffspring:
            ind.fitness.values = hybridEvolution.evaluationOnEmulator(
                ind,
                sfcrs,
                gen,
                NGEN,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
                MAX_MEMORY_DEMAND,
            )

        if len(pop) == 1:
            mutHof: tools.ParetoFront = tools.ParetoFront()
            mutHof.update(offspring + pop)
            emHof.update(offspring + pop)
            pop = mutHof

        else:
            pop, emHof = select(offspring, pop, toolbox, emPopSize, emHof)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies)
        writePFs(gen, emHof)

        emQualifiedIndividuals = [
            ind
            for ind in emHof
            if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
        ]

        emMinAR = min(ars)
        emMaxLatency = max(latencies)

        TUI.appendToSolverLog(
            f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
        )

        gen = gen + 1
