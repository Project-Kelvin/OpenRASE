"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

import os
import timeit
from typing import Callable
from uuid import uuid4, UUID
import numpy as np
from deap import base, creator, tools
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig
from algorithms.models.embedding import DecodedIndividual
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from sfc.traffic_generator import TrafficGenerator
from algorithms.ga_dijkstra_algorithm.ga_utils import (
    algorithm,
    decodePop,
    evaluation,
    generateRandomIndividual,
    mutate,
)
from models.calibrate import ResourceDemand
from utils.tui import TUI

NO_OF_INDIVIDUALS: int = 10

MAIN_DIR: str = "ga_dijkstra_algorithm"


class Individual(list):
    """
    Individual class for DEAP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id: UUID = uuid4()
        self.fitness: base.Fitness = creator.MaxARMinLatency()


def GADijkstraAlgorithm(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    dirName: str,
) -> "tuple[tools.ParetoFront]":
    """
    Run the Genetic Algorithm + Dijkstra Algorithm.

    Parameters:
        topology (Topology): the topology.
        fgrs (list[EmbeddingGraph]): the FG Requests.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.

    Returns:
        tuple[tools.ParetoFront]: the Pareto Front
    """

    TUI.appendToSolverLog(f"Starting Experiment: {dirName}")
    expDir: str = os.path.join(
        getConfig()["repoAbsolutePath"], "artifacts", "experiments"
    )
    if not os.path.exists(os.path.join(expDir, MAIN_DIR)):
        os.makedirs(os.path.join(expDir, MAIN_DIR))

    if not os.path.exists(os.path.join(expDir, MAIN_DIR, dirName)):
        os.makedirs(os.path.join(expDir, MAIN_DIR, dirName))

    dataDir: str = os.path.join(expDir, MAIN_DIR, dirName, "data.csv")
    pfDir: str = os.path.join(expDir, MAIN_DIR, dirName, "pfs.csv")
    timeDir: str = os.path.join(expDir, MAIN_DIR, dirName, "times.log")

    with open(
        dataDir,
        "w",
        encoding="utf8",
    ) as topologyFile:
        topologyFile.write(
            "generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
        )

    with open(
        pfDir,
        "w",
        encoding="utf8",
    ) as pf:
        pf.write("generation, latency, ar\n")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register("individual", generateRandomIndividual, Individual, topology, fgrs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=1.0)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=NO_OF_INDIVIDUALS)
    gen: int = 1

    CXPB, MUTPB, NGEN = 1.0, 1.0, 10
    TIME_LIMIT: int = 24 * 60 * 60

    startTime: float = timeit.default_timer()
    decodedPop: "list[DecodedIndividual]" = decodePop(pop, topology, fgrs)
    HybridEvaluation.cacheForOnline(decodedPop, trafficDesign)
    for ind in decodedPop:
        individual: "Individual" = pop[ind[0]]
        individual.fitness.values = evaluation(
            ind,
            fgrs,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
        )

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    with open(
        dataDir,
        "a",
        encoding="utf8",
    ) as topoFile:
        topoFile.write(
            f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )

    hof = tools.ParetoFront()
    hof.update(pop)

    for ind in hof:
        with open(
            pfDir,
            "a",
            encoding="utf8",
        ) as pfFile:
            pfFile.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")

    gen = gen + 1

    elapsedTime = timeit.default_timer() - startTime

    while elapsedTime < TIME_LIMIT:
        TUI.appendToSolverLog(f"Generation: {gen}")
        offspring = algorithm(pop, toolbox, CXPB, MUTPB)
        decodedOffspring: "list[DecodedIndividual]" = decodePop(
            offspring, topology, fgrs
        )
        HybridEvaluation.cacheForOnline(decodedOffspring, trafficDesign)
        for ind in decodedOffspring:
            individual: "Individual" = offspring[ind[0]]
            individual.fitness.values = evaluation(
                ind,
                fgrs,
                gen,
                NGEN,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
            )
        pop[:] = toolbox.select(pop + offspring, k=NO_OF_INDIVIDUALS)
        hof.update(pop)

        TUI.appendToSolverLog(f"Pareto Front for Generation {gen}:")
        for ind in hof:
            with open(
                pfDir,
                "a",
                encoding="utf8",
            ) as pfFile:
                pfFile.write(
                    f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n"
                )

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        TUI.appendToSolverLog(
            f"Average AR: {np.mean(ars)}, Average Latency: {np.mean(latencies)}"
        )
        with open(
            dataDir,
            "a",
            encoding="utf8",
        ) as topoFile:
            topoFile.write(
                f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
            )

        gen = gen + 1
        elapsedTime = timeit.default_timer() - startTime
        TUI.appendToSolverLog(f"Elapsed Time: {elapsedTime} seconds")

    endTime: float = timeit.default_timer()

    with open(
        timeDir,
        "w",
        encoding="utf8",
    ) as timeFile:
        timeFile.write(f"Time taken: {endTime - startTime} seconds")
    TUI.appendToSolverLog(f"Time taken: {endTime - startTime} seconds")

    return hof
