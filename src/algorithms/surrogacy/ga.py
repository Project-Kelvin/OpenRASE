"""
This defines the GA that evolves teh weights of the Neural Network.
"""

import random
import threading
from time import sleep
from timeit import default_timer
from typing import Callable, Union
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from algorithms.surrogacy.surrogate import getSFCScores
from models.traffic_generator import TrafficData
from packages.python.shared.models.embedding_graph import VNF
from packages.python.shared.models.traffic_design import TrafficDesign
from sfc.traffic_generator import TrafficGenerator
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from utils.embedding_graph import traverseVNF
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI
import os

NO_OF_WEIGHTS: int = 169 #136 weights + 22 bias for VNF embedding & 8 weights + 3 bias for link embedding.

directory = f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy"
if not os.path.exists(directory):
    os.makedirs(directory)

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "w", encoding="utf8") as topologyFile:
    topologyFile.write("generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv", "w", encoding="utf8") as pf:
    pf.write("generation, latency, ar\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/weights.csv", "w", encoding="utf8") as weights:
    weights.write("generation, w1, w2, w3, w4, w5, w6, w7, w8, w9, latency\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/latency.csv", "w", encoding="utf8") as latency:
    latency.write("sfc, reqps, cpu, memory, link, latency\n")

def evaluate(individual: "list[float]", fgs: "list[EmbeddingGraph]",  gen: int, ngen: int, sendEGs: "Callable[[list[EmbeddingGraph]], None]", deleteEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator, topology: Topology) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.

    Returns:
        tuple[float, float]: the fitness.
    """

    df: pd.DataFrame = convertFGstoDF(fgs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, individual[0:136], individual[136:158])
    egs, nodes, embedData = convertDFtoFGs(newDF, fgs, topology)

    if len(egs) > 0:
        embedLinks: EmbedLinks = EmbedLinks(topology, egs, individual[158:166], individual[166:169])
        start: float = default_timer()
        egs = embedLinks.embedLinks(nodes)
        end: float = default_timer()
        TUI.appendToSolverLog(f"Link Embedding Time for all EGs: {end - start}s")

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs)/len(fgs)
    latency: int = 0

    TUI.appendToSolverLog(f"Acceptance Ratio: {len(egs)}/{len(fgs)} = {acceptanceRatio}")

    if len(egs) > 0:
        sendEGs(egs)

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")

        step: int = 0
        interval: int = 5
        sleep(interval)
        data: "list[dict[str, dict[str, Union[int, float]]]]" = []
        while step < duration:
            try:
                sleep(interval)
                trafficData: "dict[str, TrafficData]" = trafficGenerator.getData(
                            f"{interval:.0f}s")

                for sfc, trafficData in trafficData.items():
                    requests: int = trafficData["httpReqs"]
                    req: float = round(requests / interval)
                    avgLatency: float = trafficData["averageLatency"]

                    data.append({
                        sfc: {
                            "reqps": req,
                            "latency": avgLatency
                        }
                    })

                step += interval

            except Exception as e:
                TUI.appendToSolverLog(str(e), True)

        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        rows: "list[list[Union[str, float]]]" = getSFCScores(data, topology, egs, embedData, embedLinks.getLinkData())

        for row in rows:
            with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/latency.csv", "a", encoding="utf8") as avgLatency:
                avgLatency.write(",".join([str(el) for el in row]) + "\n")

        trafficData: "dict[str, TrafficData]" = trafficGenerator.getData(
                        f"{duration:.0f}s")
        latency: float = 0
        for _key, value in trafficData.items():
            latency += value["averageLatency"]

        latency = latency / len(trafficData) if len(trafficData) > 0 else penaltyLatency

        weightRow: str = f"{gen}, "

        for weight in individual:
            weightRow += f"{weight}, "

        weightRow += f"{latency}\n"

        with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/weights.csv", "a", encoding="utf8") as weights:
            weights.write(weightRow)

        TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
        deleteEGs(egs)
    else:
        penalty: float = gen/ngen
        latency = penaltyLatency * penalty

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return acceptanceRatio, latency

def evolveWeights(fgs: "list[EmbeddingGraph]", sendEGs: "Callable[[list[EmbeddingGraph]], None]", deleteEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator, topology: Topology) -> "list[EmbeddingGraph]":
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.

    Returns:
        list[EmbeddingGraph]: the evolved Embedding Graphs.
    """

    POP_SIZE: int = 10
    NGEN: int = 10
    CXPB: float = 1.0
    MUTPB: float = 0.3

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.gauss, 0.0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=NO_OF_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1
    for ind in pop:
        ind.fitness.values = evaluate(ind, fgs, gen, NGEN, sendEGs, deleteEGs, trafficDesign, trafficGenerator, topology)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "a", encoding="utf8") as topologyFile:
        topologyFile.write(f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n")

    hof = tools.ParetoFront()
    hof.update(pop)

    for ind in hof:
        TUI.appendToSolverLog(f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}")
        with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv", "a", encoding="utf8") as pf:
            pf.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")

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
            ind.fitness.values = evaluate(ind, fgs, gen, NGEN, sendEGs, deleteEGs, trafficDesign, trafficGenerator, topology)
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        hof.update(pop)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "a", encoding="utf8") as topologyFile:
            topologyFile.write(f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n")

        for ind in hof:
            TUI.appendToSolverLog(f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}")
            with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv", "a", encoding="utf8") as pf:
                pf.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")
        gen = gen + 1
