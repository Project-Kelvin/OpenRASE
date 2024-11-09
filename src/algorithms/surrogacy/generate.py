"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
import threading
from typing import Tuple
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from algorithms.surrogacy.scorer import Scorer
from models.calibrate import ResourceDemand
from utils.tui import TUI
import tensorflow as tf
from shared.models.traffic_design import TrafficDesign
from multiprocessing import Manager, Process

tf.get_logger().setLevel('ERROR')

scorer: Scorer = Scorer()

def evaluate(individual: "list[float]", fgs: "list[EmbeddingGraph]",  gen: int, ngen: int, trafficDesign: TrafficDesign, topology: Topology) -> "tuple[float, float, float]":
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
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(individual, copiedFGs, topology)
    df: pd.DataFrame = convertFGstoDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, nodes, embedData = convertDFtoFGs(newDF, copiedFGs, topology)

    penaltyScore: float = 50
    acceptanceRatio: float = len(egs)/len(fgs)
    penalty: float = gen/ngen

    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]
    if len(egs) > 0:
        # Validate EGs
        data: "dict[str, dict[str, float]]" = {
            eg["sfcID"]: {
                "reqps": maxReqps
            } for eg in egs
        }

        scorer.cacheData(data, egs)
        scores: "dict[str, ResourceDemand]" = scorer.getHostScores(data, topology, embedData)
        maxCPU: float = max([score["cpu"] for score in scores.values()])
        maxMemory: float = max([score["memory"] for score in scores.values()])

        if acceptanceRatio < 0.75:
            maxCPU = maxCPU + (penalty * penaltyScore)
            maxMemory = maxMemory + (penalty * penaltyScore)

        return acceptanceRatio, maxCPU, maxMemory
    else:
        maxCPU: float = penalty * penaltyScore
        maxMemory: float = penalty * penaltyScore

        return acceptanceRatio, maxCPU, maxMemory


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

def evolveInitialWeights(popSize: int, fgs: "list[EmbeddingGraph]", trafficDesign: TrafficDesign, topology: Topology, maxCPU: float, maxMemory: float) -> "list[list[float]]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        popSize (int): the population size.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        maxCPU (float): The maximum CPU demand.
        maxMemory (float): The maximum memory demand.

    Returns:
        list[list[float]]: the weights.
    """

    POP_SIZE: int = popSize
    NGEN: int = 1
    CXPB: float = 1.0
    MUTPB: float = 0.8

    creator.create("MaxHosts", base.Fitness, weights=(1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxHosts)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=getWeightLength(fgs, topology))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.3)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    for ind in pop:
        ind.fitness.values = evaluate(ind, fgs, gen, NGEN, trafficDesign, topology)

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
            ind.fitness.values = evaluate(ind, fgs, gen, NGEN, trafficDesign, topology)


        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        TUI.appendToSolverLog(f"Generation {gen} completed.")
        TUI.appendToSolverLog(f"Average AR is {np.mean([ind.fitness.values[0] for ind in pop])}. Max is {max([ind.fitness.values[0] for ind in pop])}. Min is {min([ind.fitness.values[0] for ind in pop])}.")
        TUI.appendToSolverLog(f"Average max CPU is {np.mean([ind.fitness.values[1] for ind in pop])}. Max is {max([ind.fitness.values[1] for ind in pop])}. Min is {min([ind.fitness.values[1] for ind in pop])}.")
        TUI.appendToSolverLog(f"Average max memory is {np.mean([ind.fitness.values[2] for ind in pop])}. Max is {max([ind.fitness.values[2] for ind in pop])}. Min is {min([ind.fitness.values[2] for ind in pop])}.")
        gen = gen + 1

    del creator.Individual

    return pop
