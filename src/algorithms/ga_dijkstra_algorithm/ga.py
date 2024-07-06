"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

from typing import Callable
from mano.vnf_manager import VNFManager
from packages.python.shared.models.traffic_design import TrafficDesign
from packages.python.shared.utils.config import getConfig
from sfc.traffic_generator import TrafficGenerator
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from deap import base, creator, tools

from algorithms.ga_dijkstra_algorithm.utils import algorithm, evaluation, generateRandomIndividual, mutate
from models.calibrate import ResourceDemand

import numpy as np

NO_OF_INDIVIDUALS: int = 2

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/data.csv", "w", encoding="utf8") as topologyFile:
    topologyFile.write("generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n")

def GADijkstraAlgorithm(topology: Topology, resourceDemands: "dict[str, ResourceDemand]", fgrs: "list[EmbeddingGraph]", sendEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator) -> "tuple[tools.ParetoFront]":
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


    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("individual", generateRandomIndividual, creator.Individual, topology, fgrs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=NO_OF_INDIVIDUALS)

    CXPB, MUTPB, NGEN = 0.8, 0.2, 1

    gen = 0
    hof = tools.ParetoFront()

    while gen < NGEN:
        gen = gen + 1
        offspring = algorithm(pop, toolbox, CXPB, MUTPB, topology, resourceDemands, fgrs)
        pop = pop + offspring
        for ind in pop:
            ind.fitness.values = evaluation(ind, fgrs, gen, NGEN, sendEGs, trafficDesign, trafficGenerator, topology, resourceDemands)
        pop = toolbox.select(pop, k=NO_OF_INDIVIDUALS)
        hof.update(pop)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/ga_dijkstra_algorithm/data.csv", "a", encoding="utf8") as topologyFile:
            topologyFile.write(f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n")

    return hof
