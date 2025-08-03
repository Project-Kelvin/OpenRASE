"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import timeit
from typing import Callable, Tuple
from uuid import UUID, uuid4
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

artifactsDir: str = os.path.join(
    getConfig()["repoAbsolutePath"], "artifacts", "experiments", "ga_hybrid"
)

class Individual(list):
    """
    Individual class for DEAP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id: UUID = uuid4()
        self.fitness: base.Fitness = creator.MaxARMinLatency()

def decodePop(
    pop: "list[Individual]", topology: Topology, fgrs: "list[EmbeddingGraph]"
) -> "list[DecodedIndividual]":
    """
    Generate Embedding Graphs from the population.

    Parameters:
        pop (list[Individual]): the population.
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
    offspring: "list[Individual]",
    pop: "list[Individual]",
    toolbox: base.Toolbox,
    popSize: int,
    hof: tools.ParetoFront,
) -> "Tuple[list[Individual], tools.ParetoFront]":
    """
    Selection function.

    Parameters:
        offspring (list[Individual]): the offspring.
        pop (list[Individual]): the population.
        toolbox (base.Toolbox): the toolbox.
        popSize (int): the population size.
        hof (tools.ParetoFront): the hall of fame.

    Returns:
        Tuple[list[Individual], tools.ParetoFront]: the population and the hall of fame.
    """

    pop[:] = toolbox.select(pop + offspring, k=popSize)

    hof.update(pop)

    return pop, hof


def writeData(
    gen: int, ars: "list[float]", latencies: "list[float]", method: str, dir: str
) -> None:
    """
    Writes the data to the file.

    Parameters:
        gen (int): the generation.
        ars (list[float]): the acceptance ratios.
        latencies (list[float]): the latencies.
        method (str): the method used.
        dir (str): the directory to write the data to.

    Returns:
        None
    """

    with open(
        f"{dir}/data.csv",
        "a",
        encoding="utf8",
    ) as dataFile:
        dataFile.write(
            f"{method}, {gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )


def writePFs(gen: int, hof: tools.ParetoFront, method: str, dir: str) -> None:
    """
    Writes the Pareto Fronts to the file.

    Parameters:
        gen (int): the generation.
        hof (tools.ParetoFront): the hall of fame.
        method (str): the method used.
        dir (str): the directory to write the data to.

    Returns:
        None
    """

    TUI.appendToSolverLog(f"Writing Pareto Fronts for generation {gen}.")
    for ind in hof:
        with open(
            f"{dir}/pfs.csv",
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
            child1.id = uuid4()
            child2.id = uuid4()

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)

            del mutant.fitness.values

    return offspring


def geneticOperation(
    parent: list[list[list[int]]],
    pop: list[list[list[int]]],
    toolbox: base.Toolbox,
    topology: Topology,
    fgrs: list[EmbeddingGraph],
    trafficDesign: list[TrafficDesign],
    dirName: str,
    scoresDir: str,
    gen: int,
    ngen: int,
    maxMemoryDemand: float,
    minAR: float,
    maxLatency: float,
    minQualifiedInds: int,
    popSize: int,
    trafficGenerator: TrafficGenerator,
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    hof: tools.ParetoFront
) -> "tuple[list[list[list[int]]], list[list[list[int]]]]":
    """
    Perform the genetic operation.

    Parameters:
        parent (list[list[list[int]]]): the parent population.
        pop (list[list[list[int]]]): the current population.
        toolbox (base.Toolbox): the toolbox.
        topology (Topology): the topology.
        fgrs (list[EmbeddingGraph]): the Forwarding Graph Requests.
        trafficDesign (list[TrafficDesign]): the traffic design.
        dirName (str): the directory name to save results.
        scoresDir (str): the directory name for scores.
        gen (int): the current generation number.
        ngen (int): the total number of generations.
        maxMemoryDemand (float): maximum memory demand allowed.
        minAR (float): minimum acceptance ratio required.
        maxLatency (float): maximum latency allowed.
        minQualifiedInds (int): minimum number of qualified individuals required.
        popSize (int): size of the population.
        trafficGenerator (TrafficGenerator): traffic generator instance.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): function to send Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): function to delete Embedding Graphs.
        hof (tools.ParetoFront): hall of fame for storing best individuals.

    Returns:
        tuple[list[list[list[int]]], list[list[list[int]]]]: the updated population and qualified individuals.
    """

    populationEG: "list[DecodedIndividual]" = decodePop(pop, topology, fgrs)
    HybridEvolution.cacheForOffline(
        populationEG, trafficDesign, topology, gen, isAvgOnly=True
    )
    HybridEvolution.saveCachedLatency(
        os.path.join(dirName, scoresDir, f"gen_{gen}.csv")
    )

    startTime: int = timeit.default_timer()
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                HybridEvolution.evaluationOnSurrogate,
                ind,
                gen,
                ngen,
                topology,
                trafficDesign,
                maxMemoryDemand,
            )
            for ind in populationEG
        ]

        for future in as_completed(futures):
            result: "tuple[int, float, float]" = future.result()
            ind: "Individual" = pop[result[0]]
            ind.fitness.values = (result[1], result[2])

    endTime: int = timeit.default_timer()
    TUI.appendToSolverLog(f"Finished generation {gen} in {endTime - startTime} seconds.")
    if len(parent) > 0:
        pop, hof = select(pop, parent, toolbox, popSize, hof)
    else:
        hof.update(pop)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies, "surrogate", dirName)
    writePFs(gen, hof, "surrogate", dirName)

    qualifiedIndividuals = [
        ind
        for ind in hof
        if ind.fitness.values[0] >= minAR and ind.fitness.values[1] <= maxLatency
    ]

    TUI.appendToSolverLog(
        f"Qualified Individuals: {len(qualifiedIndividuals)}/{minQualifiedInds}"
    )

    if len(qualifiedIndividuals) >= minQualifiedInds:
        TUI.appendToSolverLog(
            f"Finished the evolution of weights using surrogate at generation {gen}."
        )
        TUI.appendToSolverLog(
            f"Number of qualified individuals: {len(qualifiedIndividuals)}"
        )

        # ---------------------------------------------------------------------------------------------
        # Start the online phase of the hybrid evolution
        # ---------------------------------------------------------------------------------------------

        for ind in qualifiedIndividuals:
            del ind.fitness.values

        emHof = tools.ParetoFront()

        populationEG: "list[DecodedIndividual]" = decodePop(
            qualifiedIndividuals, topology, fgrs
        )
        HybridEvolution.cacheForOnline(populationEG, trafficDesign)
        for ind in populationEG:
            ar, latency = HybridEvolution.evaluationOnEmulator(
                ind,
                fgrs,
                gen,
                ngen,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
                maxMemoryDemand,
            )
            qualifiedIndividuals[ind[0]].fitness.values = (ar, latency)

            for p in pop:
                if p.id == qualifiedIndividuals[ind[0]].id:
                    p.fitness.values = (ar, latency)
                    break

        emHof.update(qualifiedIndividuals)

        ars = [ind.fitness.values[0] for ind in qualifiedIndividuals]
        latencies = [ind.fitness.values[1] for ind in qualifiedIndividuals]

        writeData(gen, ars, latencies, "emulator", dirName)
        writePFs(gen, emHof, "emulator", dirName)

        qualifiedIndividuals = [
            ind
            for ind in emHof
            if ind.fitness.values[0] >= minAR
            and ind.fitness.values[1] <= maxLatency
        ]

        emMinAR = min(ars)
        emMaxLatency = max(latencies)

        TUI.appendToSolverLog(
            f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
        )

    return pop, qualifiedIndividuals


def hybridSolver(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: "list[TrafficDesign]",
    trafficGenerator: TrafficGenerator,
    experiment: str
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
        experiment (str): the experiment name.

    Returns:
        tuple[tools.ParetoFront]: the Pareto Front
    """

    TUI.appendToSolverLog(
        f"Running the hybrid online-offline solver for experiment: {experiment}"
    )

    expStartTime: int = timeit.default_timer()
    POP_SIZE: int = 2000
    NGEN = 500
    MAX_MEMORY_DEMAND: int = 2
    MAX_LATENCY: int = 150
    MIN_AR: float = 1.0
    MIN_QUAL_IND: int = 1
    CXPB: float = 1.0
    MUTPB: float = 1.0
    SCORES_DIR: str = "scores"

    expDir: str = os.path.join(
        artifactsDir, experiment
    )

    if not os.path.exists(expDir):
        os.makedirs(expDir)

    if not os.path.exists(os.path.join(expDir, "scores")):
        os.makedirs(os.path.join(expDir, "scores"))

    with open(
        os.path.join(
            expDir, "data.csv"
        ),
        "w",
        encoding="utf8",
    ) as topologyFile:
        topologyFile.write(
            "method, generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
        )

    with open(
        os.path.join(
            expDir, "pfs.csv"
        ),
        "w",
        encoding="utf8",
    ) as pf:
        pf.write("method, generation, latency, ar\n")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register(
        "individual", generateRandomIndividual, Individual, topology, fgrs
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate, indpb=1.0)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    hof: tools.ParetoFront = tools.ParetoFront()
    pop, qualifiedIndividuals = geneticOperation(
        [],
        pop,
        toolbox,
        topology,
        fgrs,
        trafficDesign,
        expDir,
        SCORES_DIR,
        gen,
        NGEN,
        MAX_MEMORY_DEMAND,
        MIN_AR,
        MAX_LATENCY,
        MIN_QUAL_IND,
        POP_SIZE,
        trafficGenerator,
        sendEGs,
        deleteEGs,
        hof
    )

    gen = gen + 1

    while len(qualifiedIndividuals) < MIN_QUAL_IND and gen <= NGEN:
        offspring: "list[Individual]" = genOffspring(toolbox, pop, CXPB, MUTPB)
        pop, qualifiedIndividuals = geneticOperation(
            pop,
            offspring,
            toolbox,
            topology,
            fgrs,
            trafficDesign,
            expDir,
            SCORES_DIR,
            gen,
            NGEN,
            MAX_MEMORY_DEMAND,
            MIN_AR,
            MAX_LATENCY,
            MIN_QUAL_IND,
            POP_SIZE,
            trafficGenerator,
            sendEGs,
            deleteEGs,
            hof
        )
        gen = gen + 1

    expEndTime: int = timeit.default_timer()
    TUI.appendToSolverLog(f"Time taken: {expEndTime - expStartTime:.2f}s")

    names: list[str] = experiment.split("_")
    with open(
        os.path.join(
            expDir, "experiment.txt"
        ),
        "w",
        encoding="utf8",
    ) as expFile:
        expFile.write(f"No. of SFCRs: {4 * int(names[0])}\n")
        expFile.write(f"Traffic Scale: {float(names[1]) * 10}\n")
        expFile.write(f"Traffic Pattern: {'Pattern B' if names[2] == 'True' else 'Pattern A'}\n")
        expFile.write(f"Link Bandwidth: {int(names[3])}\n")
        expFile.write(f"No. of CPUs: {int(names[4])}\n")
        expFile.write(f"Time taken: {expEndTime - expStartTime:.2f}\n")
