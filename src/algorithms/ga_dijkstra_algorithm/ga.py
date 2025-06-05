"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

import timeit
from typing import Callable
import numpy as np
from deap import base, creator, tools
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig
from sfc.traffic_generator import TrafficGenerator
from algorithms.ga_dijkstra_algorithm.utils import (
    algorithm,
    crossover,
    evaluation,
    generateRandomIndividual,
    getVNFsfromFGRs,
    mutate,
)
from models.calibrate import ResourceDemand
from utils.tui import TUI

NO_OF_INDIVIDUALS: int = 10

with open(
    f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/data.csv",
    "w",
    encoding="utf8",
) as topologyFile:
    topologyFile.write(
        "generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
    )

with open(
    f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/pfs.csv",
    "w",
    encoding="utf8",
) as pf:
    pf.write("generation, latency, ar\n")


def GADijkstraAlgorithm(
    topology: Topology,
    resourceDemands: "dict[str, ResourceDemand]",
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
) -> "tuple[tools.ParetoFront]":
    """
    Run the Genetic Algorithm + Dijkstra Algorithm.

    Parameters:
        topology (Topology): the topology.
        resourceDemands (dict[str, ResourceDemand]): the resource demands.
        fgrs (list[EmbeddingGraph]): the FG Requests.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.

    Returns:
        tuple[tools.ParetoFront]: the Pareto Front
    """

    startTime: float = timeit.default_timer()
    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register(
        "individual", generateRandomIndividual, creator.Individual, topology, fgrs
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=1.0)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=NO_OF_INDIVIDUALS)
    gen: int = 1

    CXPB, MUTPB, NGEN = 1.0, 1.0, 10

    for ind in pop:
        ind.fitness.values = evaluation(
            ind,
            fgrs,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
            resourceDemands,
        )

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    with open(
        f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/data.csv",
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
            f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/pfs.csv",
            "a",
            encoding="utf8",
        ) as pfFile:
            pfFile.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")

    gen = gen + 1

    while gen <= NGEN:
        TUI.appendToSolverLog(f"Generation: {gen}")
        offspring = algorithm(pop, toolbox, CXPB, MUTPB)
        for ind in offspring:
            ind.fitness.values = evaluation(
                ind,
                fgrs,
                gen,
                NGEN,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
                resourceDemands,
            )
        pop[:] = toolbox.select(pop + offspring, k=NO_OF_INDIVIDUALS)
        hof.update(pop)

        for ind in hof:
            with open(
                f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/pfs.csv",
                "a",
                encoding="utf8",
            ) as pfFile:
                pfFile.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        with open(
            f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/data.csv",
            "a",
            encoding="utf8",
        ) as topoFile:
            topoFile.write(
                f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
            )

        gen = gen + 1

    endTime: float = timeit.default_timer()
    TUI.appendToSolverLog(f"Time taken: {endTime - startTime} seconds")

    return hof
