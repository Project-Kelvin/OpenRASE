"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import timeit
from typing import Any, Callable, Tuple
from deap import base, creator, tools
import numpy as np
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from shared.utils.config import getConfig
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from algorithms.ga_dijkstra_algorithm.ga_utils import (
    convertIndividualToEmbeddingGraph,
    generateRandomIndividual,
    mutate,
)
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI


with open(
    f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv",
    "w",
    encoding="utf8",
) as topologyFile:
    topologyFile.write(
        "method, generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
    )

with open(
    f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv",
    "w",
    encoding="utf8",
) as pf:
    pf.write("method, generation, latency, ar\n")


def decodePop(
    pop: "list[creator.Individual]", topology: Topology, fgrs: "list[EmbeddingGraph]"
) -> "list[DecodedIndividual]":
    """
    Generate Embedding Graphs from the population.

    Parameters:
        pop (list[creator.Individual]): the population.
        topology (Topology): the topology.
        fgrs (list[EmbeddingGraph]): the Forwarding Graph Requests.

    Returns:
        list[IndividualEG]: A list containing EGs, embedding data, link data and acceptance ratio.
    """

    populationEG: "list[DecodedIndividual]" = []

    for index, individual in enumerate(pop):
        egs, embeddingData, linkData = convertIndividualToEmbeddingGraph(
            individual, fgrs, topology
        )

        acceptanceRatio: float = len(egs) / len(fgrs)

        populationEG.append((index, egs, embeddingData, linkData, acceptanceRatio))

    return populationEG

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


def writeData(
    gen: int, ars: "list[float]", latencies: "list[float]", method: str
) -> None:
    """
    Writes the data to the file.

    Parameters:
        gen (int): the generation.
        ars (list[float]): the acceptance ratios.
        latencies (list[float]): the latencies.
        method (str): the method used.

    Returns:
        None
    """

    with open(
        f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv",
        "a",
        encoding="utf8",
    ) as dataFile:
        dataFile.write(
            f"{method}, {gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )


def writePFs(gen: int, hof: tools.ParetoFront, method: str) -> None:
    """
    Writes the Pareto Fronts to the file.

    Parameters:
        gen (int): the generation.
        hof (tools.ParetoFront): the hall of fame.
        method (str): the method used.

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
            pfFile.write(
                f"{method}, {gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n"
            )


def genOffspring(
    toolbox: base.Toolbox, pop: "list[list[list[int]]]", CXPB: float, MUTPB: float
) -> "list[list[list[int]]]":
    """
    Generate offspring from the population.

    Parameters:
        toolbox (base.Toolbox): the toolbox.
        pop (list[list[list[int]]]): the population.
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

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

            del mutant.fitness.values

    return offspring


def hybridSolver(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: "list[TrafficDesign]",
    trafficGenerator: TrafficGenerator,
) -> "tuple[tools.ParetoFront]":
    """
    Run the Genetic Algorithm + Dijkstra Algorithm.

    Parameters:
        topology (Topology): the topology.
        resourceDemands (dict[str, ResourceDemand]): the resource demands.
        fgrs (list[EmbeddingGraph]): the FG Requests.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        trafficDesign (list[TrafficDesign]): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.

    Returns:
        tuple[tools.ParetoFront]: the Pareto Front
    """

    startTime: int = timeit.default_timer()
    POP_SIZE: int = 100
    NGEN = 100
    MAX_MEMORY_DEMAND: int = 2
    MAX_LATENCY: int = 500
    MIN_AR: float = 1
    MIN_QUAL_IND: int = 1
    CXPB: float = 1.0
    MUTPB: float = 1.0

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register(
        "individual", generateRandomIndividual, creator.Individual, topology, fgrs
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate, indpb=1.0)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    startTime: int = timeit.default_timer()
    populationEG: "list[DecodedIndividual]" = decodePop(pop, topology, fgrs)
    HybridEvolution.cacheForOffline(populationEG, trafficDesign, topology, gen)
    TUI.appendToSolverLog("Population demand and traffic latency cached.")
    for ind in populationEG:
        index, ar, latency = HybridEvolution.evaluationOnSurrogate(
            ind,
            gen,
            NGEN,
            topology,
            trafficDesign,
            MAX_MEMORY_DEMAND,
        )
        pop[index].fitness.values = (ar, latency)
    # with ProcessPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(
    #             HybridEvolution.evaluationOnSurrogate,
    #             ind,
    #             gen,
    #             NGEN,
    #             topology,
    #             trafficDesign,
    #             MAX_MEMORY_DEMAND,
    #         )
    #         for ind in populationEG
    #     ]
    #     for future in as_completed(futures):
    #         result: "tuple[int, float, float]" = future.result()
    #         ind: "creator.Individual" = pop[result[0]]
    #         ind.fitness.values = (result[1], result[2])

    endTime: int = timeit.default_timer()
    print("First generation time: ", endTime - startTime)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies, "surrogate")

    hof = tools.ParetoFront()
    hof.update(pop)

    writePFs(gen, hof, "surrogate")

    gen = gen + 1

    qualifiedIndividuals: "list[creator.Individual]" = [
        ind
        for ind in hof
        if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
    ]

    while len(qualifiedIndividuals) < MIN_QUAL_IND:
        offspring: "list[creator.Individual]" = genOffspring(toolbox, pop, CXPB, MUTPB)

        populationEG: "list[DecodedIndividual]" = decodePop(offspring, topology, fgrs)
        startTime: int = timeit.default_timer()
        HybridEvolution.cacheForOffline(populationEG, trafficDesign, topology, gen)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    HybridEvolution.evaluationOnSurrogate,
                    ind,
                    gen,
                    NGEN,
                    topology,
                    trafficDesign,
                    MAX_MEMORY_DEMAND,
                )
                for ind in populationEG
            ]

            for future in as_completed(futures):
                result: "tuple[int, float, float]" = future.result()
                ind: "creator.Individual" = offspring[result[0]]
                ind.fitness.values = (result[1], result[2])

        endTime: int = timeit.default_timer()
        print("Finished generation ", gen, " in ", endTime - startTime, " seconds.")
        pop, hof = select(offspring, pop, toolbox, POP_SIZE, hof)
        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies, "surrogate")
        writePFs(gen, hof, "surrogate")

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
    # Start the online phase of the hybrid evolution
    # ---------------------------------------------------------------------------------------------

    pop = list(qualifiedIndividuals)

    for ind in pop:
        del ind.fitness.values

    emHof = tools.ParetoFront()
    emPopSize = len(pop)
    emQualifiedIndividuals: "list[creator.Individual]" = []
    EM_MIN_QUAL_IND: int = 1

    populationEG: "list[DecodedIndividual]" = decodePop(pop, topology, fgrs)
    HybridEvolution.cacheForOnline(populationEG, trafficDesign)
    for ind in populationEG:
        ar, latency = HybridEvolution.evaluationOnEmulator(
            ind,
            fgrs,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
            MAX_MEMORY_DEMAND,
        )
        pop[ind[0]].fitness.values = (ar, latency)

    emHof.update(pop)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies, "emulator")
    writePFs(gen, emHof, "emulator")

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
            offspring = [mutant]
        else:
            offspring = genOffspring(toolbox, pop, CXPB, MUTPB)

        offspringEG: "list[DecodedIndividual]" = decodePop(offspring, topology, fgrs)
        HybridEvolution.cacheForOnline(offspringEG, trafficDesign)
        for ind in offspringEG:
            geneticIndividual: creator.Individual = offspring[ind[0]]
            if (
                hasattr(geneticIndividual.fitness, "values")
                and len(geneticIndividual.fitness.values) > 0
            ):
                TUI.appendToSolverLog("Individual already evaluated.")
            else:
                geneticIndividual.fitness.values = HybridEvolution.evaluationOnEmulator(
                    ind,
                    fgrs,
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
            pop = list(mutHof)
        else:
            pop, emHof = select(offspring, pop, toolbox, emPopSize, emHof)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies, "emulator")
        writePFs(gen, emHof, "emulator")

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

    endTime: int = timeit.default_timer()
    TUI.appendToSolverLog(f"Time taken: {endTime - startTime:.2f}s")
