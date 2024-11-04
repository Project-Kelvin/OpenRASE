"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
from typing import Tuple
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from utils.tui import TUI
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

def evaluate(individual: "list[float]", fgs: "list[EmbeddingGraph]", topology: Topology) -> float:
    """
    Evaluates the individual.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        float: the acceptance ratio.
    """

    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(individual, copiedFGs, topology)
    df: pd.DataFrame = convertFGstoDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    _egs, _nodes, embedData = convertDFtoFGs(newDF, copiedFGs, topology)

    return len(embedData)


def getLinkWeight(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of link weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of link weights.
    """

    return len(fgs) + 2 * (len(topology["hosts"]) + len(topology["switches"]) + 2)

def getVNFWeight(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of VNF weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of VNF weights.
    """

    return len(fgs) + len(getConfig()["vnfs"]["names"]) + len(topology["hosts"]) + 1

def getLinkBias() -> int:
    """
    Gets the number of link biases.

    Returns:
        int: the number of link biases.
    """

    return 1

def getVNFBias() -> int:
    """
    Gets the number of VNF biases.

    Returns:
        int: the number of VNF biases.
    """

    return 1

def getWeightLength(fgs: "list[EmbeddingGraph]", topology: Topology) -> int:
    """
    Gets the number of weights.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        int: the number of weights.
    """

    return getLinkWeight(fgs, topology) + getVNFWeight(fgs, topology) + getLinkBias() + getVNFBias()

def getWeights(individual: "list[float]", fgs: "list[EmbeddingGraph]", topology: Topology) -> "Tuple[list[float], list[float], list[float], list[float]]":
    """
    Gets the weights.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: VNF weights, VNF bias, link weights, link bias.
    """

    vnfWeights: int = getVNFWeight(fgs, topology)
    linkWeights: int = getLinkWeight(fgs, topology)
    vnfBias: int = getVNFBias()
    linkBias: int = getLinkBias()

    vnfWeightUpper: int = vnfWeights
    vnfBiasUpper: int = vnfWeights + vnfBias
    linkWeightUpper: int = vnfWeights + vnfBias + linkWeights
    linkBiasUpper: int = vnfWeights + vnfBias + linkWeights + linkBias

    return individual[0:vnfWeightUpper], individual[vnfWeightUpper:vnfBiasUpper], individual[vnfBiasUpper:linkWeightUpper], individual[linkWeightUpper:linkBiasUpper]

def evolveInitialWeights(fgs: "list[EmbeddingGraph]", topology: Topology) -> "list[list[float]]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        topology (Topology): the topology.

    Returns:
        list[list[float]]: the weights.
    """

    POP_SIZE: int = 500
    NGEN: int = 100
    CXPB: float = 1.0
    MUTPB: float = 1.0

    creator.create("MaxHosts", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.MaxHosts)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -100, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=getWeightLength(fgs, topology))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=10000.0, indpb=1.09)
    toolbox.register("select", tools.selTournament)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    for ind in pop:
        ind.fitness.values = (evaluate(ind, fgs, topology),)

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
            ind.fitness.values = (evaluate(ind, fgs, topology),)
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE, tournsize=POP_SIZE//100)


        TUI.appendToSolverLog(f"Generation {gen} completed. Average is {np.mean([ind.fitness.values[0] for ind in pop])}. Max is {max([ind.fitness.values[0] for ind in pop])}. Min is {min([ind.fitness.values[0] for ind in pop])}.")
        gen = gen + 1

    return pop
