"""
This defines the GA that evolves teh weights of the Neural Network.
"""

import json
import random
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from utils.embedding_graph import traverseVNF
from utils.topology import generateFatTreeTopology

def evaluate(individual: "list[float]", fgs: "list[EmbeddingGraph]", topology: Topology) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        tuple[float, float]: the fitness.
    """

    df: pd.DataFrame = convertFGstoDF(fgs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, individual[0:-1], [individual[-1]])
    egs: "list[EmbeddingGraph]" = convertDFtoFGs(newDF, fgs, topology)

    ar: float = len(egs)/len(fgs)
    latency: int = 0
    hosts: "dict[str, int]" = {}
    for eg in egs:
        def parseVNF(vnf: VNF, _depth: int, hosts: "dict[str, int]") -> None:
            """
            Parses a VNF.

            Parameters:
                vnf (VNF): the VNF.
                _depth (int): the depth.
                hosts (dict[str, int]): the hosts.
            """

            if vnf["host"]["id"] in hosts:
                hosts[vnf["host"]["id"]] = hosts[vnf["host"]["id"]] + 1
            else:
                hosts[vnf["host"]["id"]] = 1

        traverseVNF(eg["vnfs"], parseVNF, hosts, shouldParseTerminal=False)

    if len(hosts.values()) > 0:
        latency = max(hosts.values())
    else:
        latency = 17

    return ar, latency

def evolveWeights(fgs: "list[EmbeddingGraph]", topology: Topology) -> "list[EmbeddingGraph]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        list[EmbeddingGraph]: the evolved Embedding Graphs.
    """

    NO_OF_WEIGHTS: int = 5 #4 weights and 1 bias
    POP_SIZE: int = 100
    NGEN: int = 5
    CXPB: float = 0.8
    MUTPB: float = 0.2

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, 0.0, 1000.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=NO_OF_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1

    for ind in pop:
        ind.fitness.values = evaluate(ind, fgs, topology)

    hof = tools.ParetoFront()
    hof.update(pop)

    for ind in hof:
        print(f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}")

    gen = gen + 1
    while gen <= NGEN:
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

        for ind in offspring:
            ind.fitness.values = evaluate(ind, fgs, topology)
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        hof.update(pop)

        for ind in hof:
            print(f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}")
        gen = gen + 1

topo: Topology = generateFatTreeTopology(4, 1000, 2, 1000)

with open(f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs/forwarding-graphs.json", "r") as file:
    fgs: "list[EmbeddingGraph]" = json.load(file)

    evolveWeights(fgs, topo)
