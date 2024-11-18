"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
import random
from time import sleep
from timeit import default_timer
from typing import Callable, Tuple, Union
from algorithms.surrogacy.extract_weights import getWeightLength
from algorithms.surrogacy.generate import evolveInitialWeights, getWeights
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.nn import convertDFtoFGs, convertFGstoDF, getConfidenceValues
from algorithms.surrogacy.scorer import Scorer
from algorithms.surrogacy.surrogate import Surrogate
from algorithms.surrogacy.surrogate_evolver import evolveUsingSurrogate
from models.calibrate import ResourceDemand
from models.traffic_generator import TrafficData
from shared.models.traffic_design import TrafficDesign
from sfc.traffic_generator import TrafficGenerator
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
import pandas as pd
import numpy as np
from deap import base, creator, tools
from shared.utils.config import getConfig
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI
import os
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.keras.utils.disable_interactive_logging()

directory = f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy"
if not os.path.exists(directory):
    os.makedirs(directory)

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "w", encoding="utf8") as topologyFile:
    topologyFile.write("generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv", "w", encoding="utf8") as pf:
    pf.write("generation, latency, ar\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/weights.csv", "w", encoding="utf8") as weights:
    weights.write("generation, w1, w2, w3, w4, w5, w6, w7, w8, w9, latency\n")

with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/latency.csv", "w", encoding="utf8") as latencyFile:
    latencyFile.write("generation,individual,sfc,reqps,cpu,avg_cpu,memory,avg_memory,link,latency,sfc_hosts,total_hosts,no_sfcs,ar\n")

scorer: Scorer = Scorer()

def evaluate(index: int, individual: "list[float]", fgs: "list[EmbeddingGraph]",  gen: int, ngen: int, sendEGs: "Callable[[list[EmbeddingGraph]], None]", deleteEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator, topology: Topology, trafficType: bool, maxCPUDemand: float, maxMemoryDemand: float) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        index (int): individual index.
        individual (list[float]): the individual.
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        trafficType (bool): whether to use the minimal traffic design.
        maxCPUDemand (float): maximum CPU demand.
        maxMemoryDemand (float): maximum memory demand.

    Returns:
        tuple[float, float]: the fitness.
    """

    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(individual, copiedFGs, topology)
    df: pd.DataFrame = convertFGstoDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, nodes, embedData = convertDFtoFGs(newDF, copiedFGs, topology)
    if len(egs) > 0:
        embedLinks: EmbedLinks = EmbedLinks(topology, egs,weights[2], weights[3])
        start: float = default_timer()
        egs = embedLinks.embedLinks(nodes)
        end: float = default_timer()
        TUI.appendToSolverLog(f"Link Embedding Time for all EGs: {end - start}s")

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs)/len(fgs)
    latency: int = 0
    penalty: float = gen/ngen
    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]
    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(fgs)} = {acceptanceRatio}"
    )
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
        TUI.appendToSolverLog(f"Max CPU: {maxCPU}, Max Memory: {maxMemory}")
        # Validate EGs
        # The resource demand of deployed VNFs exceeds the resource capacity of at least 1 host.
        # This leads to servers crashing.
        # Penalty is applied to the latency and the egs are not deployed.
        if maxCPU > maxCPUDemand or maxMemory > maxMemoryDemand:
            TUI.appendToSolverLog(f"Penalty because max CPU demand is {maxCPU} and max Memory demand is {maxMemory}.")
            latency = penaltyLatency * penalty * (maxCPU + maxMemory)
            acceptanceRatio = acceptanceRatio - (penalty * (maxCPU + maxMemory))

            return acceptanceRatio, latency

        reqs: "list[int]" = sum([int(td["target"]) * int(td["duration"][:-1]) for td in trafficDesign[0]])
        time: "list[int]" = sum([int(td["duration"][:-1]) for td in trafficDesign[0]])
        avg: float =  reqs / time

        data: "dict[str, dict[str, float]]" = {
            eg["sfcID"]: {
                "reqps": avg,
                "latency": 0
            } for eg in egs
        }

        rows: "list[list[Union[str, float]]]" = scorer.getSFCScores(data, topology, egs, embedData, embedLinks.getLinkData())

        for row in rows:
            row.append(len(egs))

        inputData: pd.DataFrame = pd.DataFrame(
            rows,
            columns=[
                "sfc",
                "reqps",
                "cpu",
                "avg_cpu",
                "memory",
                "avg_memory",
                "link",
                "latency",
                "sfc_hosts",
                "no_sfcs"
            ],
        )

        surrogate: Surrogate = Surrogate()
        outputData: pd.DataFrame = surrogate.predict(inputData)

        latency = outputData["PredictedLatency"].mean()
        confidence: float = outputData["Confidence"].mean()

        if acceptanceRatio >= 0.8 and latency <= 1500 and confidence > 50:
            sendEGs(egs)

            duration: int = calculateTrafficDuration(trafficDesign[0])
            TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
            TUI.appendToSolverLog(f"Waiting for {duration}s...")

            sleep(duration)

            TUI.appendToSolverLog(f"Done waiting for {duration}s.")

            anomalousDuration: int = 15 if not trafficType else 2
            trafficDuration: int = duration - anomalousDuration
            trafficData: "dict[str, TrafficData]" = trafficGenerator.getData(
                            f"{trafficDuration:.0f}s")
            latency: float = 0
            for key, value in trafficData.items():
                latency += value["averageLatency"]

                outputData.loc[outputData["sfc"] == key, "latency"] = value["averageLatency"]

            for row in rows:
                with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/latency.csv", "a", encoding="utf8") as avgLatency:
                    avgLatency.write(f"{gen},{index}," + ",".join([str(el) for el in row]) + f",{len(scores)},{len(egs)},{acceptanceRatio}\n")

            latency = latency / len(trafficData) if len(trafficData) > 0 else penaltyLatency

            weightRow: str = f"{gen}, "

            for weight in individual:
                weightRow += f"{weight}, "

            weightRow += f"{latency}\n"

            with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/weights.csv", "a", encoding="utf8") as weights:
                weights.write(weightRow)

            trainData: pd.DataFrame = outputData.loc[outputData["latency"] > 0]
            if len(trainData) > 0:
                surrogate.trainModel(trainData)

            TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
            deleteEGs(egs)
    else:
        latency = penaltyLatency * penalty

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return acceptanceRatio, latency


def evolveWeights(fgs: "list[EmbeddingGraph]", sendEGs: "Callable[[list[EmbeddingGraph]], None]", deleteEGs: "Callable[[list[EmbeddingGraph]], None]", trafficDesign: TrafficDesign, trafficGenerator: TrafficGenerator, topology: Topology, trafficType: bool) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        trafficType (bool): whether to use the minimal traffic design.

    Returns:
        None
    """

    POP_SIZE: int = 10
    NGEN: int = 5
    CXPB: float = 1.0
    MUTPB: float = 0.8
    maxCPUDemand: int = 1
    maxMemoryDemand: int = 5

    evolvedPop: "list[creator.Individual]" = evolveUsingSurrogate(fgs, trafficDesign, topology, POP_SIZE)
    print(len(evolvedPop))
    TUI.appendToSolverLog("Starting the evolution of the weights using OpenRASE.")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    evolvedNewPop: "list[creator.Individual]" = []
    for ep in evolvedPop:
        ind = creator.Individual()
        ind.extend(ep)
        evolvedNewPop.append(ind)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=getWeightLength(fgs, topology))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.8)
    toolbox.register("select", tools.selNSGA2)

    randomPop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    alpha: float = 1.0
    pop: "list[creator.Individual]" = random.sample(evolvedNewPop, int(POP_SIZE * alpha)) + random.sample(randomPop, int(POP_SIZE * (1 - alpha)))
    gen: int = 1
    for i, ind in enumerate(pop):
        ind.fitness.values = evaluate(i, ind, fgs, gen, NGEN, sendEGs, deleteEGs, trafficDesign, trafficGenerator, topology, trafficType, maxCPUDemand, maxMemoryDemand)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "a", encoding="utf8") as dataFile:
        dataFile.write(f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n")

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

        for i, ind in enumerate(offspring):
            ind.fitness.values = evaluate(i, ind, fgs, gen, NGEN, sendEGs, deleteEGs, trafficDesign, trafficGenerator, topology, trafficType, maxCPUDemand, maxMemoryDemand)
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        hof.update(pop)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv", "a", encoding="utf8") as dataFile:
            dataFile.write(f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n")

        for ind in hof:
            TUI.appendToSolverLog(f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}")
            with open(f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv", "a", encoding="utf8") as pf:
                pf.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")
        gen = gen + 1
