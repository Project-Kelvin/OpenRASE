"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from math import floor
import random
from time import sleep
import os
from typing import Callable
import pandas as pd
import polars as pl
from deap import base, creator, tools
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.utils.extract_weights import getWeightLength
from algorithms.surrogacy.constants.surrogate import SURROGACY_PATH, SURROGATE_DATA_PATH, SURROGATE_PATH
from algorithms.surrogacy.hybrid_online_offline import decodePop
from algorithms.surrogacy.surrogate.data_generator.init_pop_generator import (
    evolveInitialWeights,
)
from algorithms.surrogacy.utils.hybrid_evolution import HybridEvolution
from algorithms.surrogacy.utils.scorer import Scorer
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI

directory: str = SURROGACY_PATH
if not os.path.exists(directory):
    os.makedirs(directory)

surrogateDirectory: str = SURROGATE_PATH
if not os.path.exists(surrogateDirectory):
    os.makedirs(surrogateDirectory)

surrogateDataDirectory: str = SURROGATE_DATA_PATH
if not os.path.exists(surrogateDataDirectory):
    os.makedirs(surrogateDataDirectory)

isFirstSetWritten: bool = False
hybridEvolution: "HybridEvolution" = HybridEvolution()

def evaluate(
    individual: DecodedIndividual,
    gen: int,
    ngen: int,
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    maxMemoryDemand: float,
    minAR: float,
    fileName: str
) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        individual (DecodedIndividual): the decoded individual.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        maxMemoryDemand (float): maximum memory demand.
        minAR (float): minimum acceptance ratio.
        fileName (str): the file name.

    Returns:
        tuple[float, float]: the fitness.
    """

    global isFirstSetWritten
    _individualIndex, egs, embedData, _linkData, acceptanceRatio = individual

    penaltyLatency: float = 50000
    penaltyWeight: float = gen / ngen
    latency: int = 0

    TUI.appendToSolverLog(
        f"Acceptance Ratio: {acceptanceRatio}"
    )

    if len(egs) > 0:
        _maxCPU, maxMemory = hybridEvolution.getMaxCpuMemoryUsageOfHosts(egs, topology, embedData, trafficDesign, maxMemoryDemand)
        if  maxMemory> maxMemoryDemand:
            TUI.appendToSolverLog(
                f"Penalty because max Memory demand is {maxMemory}."
            )
            latency = penaltyLatency * penaltyWeight * (maxMemory)
            acceptanceRatio = acceptanceRatio - (penaltyWeight * (maxMemory))

            return acceptanceRatio, latency

        # deploy
        sendEGs(egs)

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")

        sleep(duration)

        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        # process traffic data
        trafficData: pd.DataFrame = trafficGenerator.getData(f"{duration + 5:.0f}s")

        if (
            trafficData.empty
            or "_time" not in trafficData.columns
            or "_value" not in trafficData.columns
        ):
            TUI.appendToSolverLog("Traffic data is empty.")

            return 0, penaltyLatency

        hybridEvolution.cacheForOffline(
            [individual],
            trafficDesign,
            topology,
            gen
        )
        data: pl.DataFrame = hybridEvolution.generateScoresForRealTrafficData(
            individual,
            trafficData,
            trafficDesign,
            topology
        )

        with open(
            f"{surrogateDirectory}/{fileName}",mode="a", encoding="utf8"
        ) as scoreFile:
            data.write_csv(
                scoreFile,
                include_header=not isFirstSetWritten,
                separator=",",
            )

        isFirstSetWritten = True

        data = data.with_columns(pl.when(pl.col("latency") == 0).then(1500).otherwise(pl.col("latency")).alias("latency"))

        latency = pl.Series(data.select("latency")).mean()

        TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
        deleteEGs(egs)
        sleep(30)
    else:
        latency = penaltyLatency * penaltyWeight

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    if acceptanceRatio < minAR:
        latency = latency + (
            penaltyLatency * penaltyWeight * (1 - floor(random.uniform(0, 1)))
        )

    return acceptanceRatio, latency


def evolveWeights(
    fgs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    dataType: int
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        fgs (list[EmbeddingGraph]): the list of Embedding Graphs.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        dataType (int): the data type (LC LL:0, LC HL:1, HC LL:2, HC HL:3).

    Returns:
        None
    """

    POP_SIZE: int = 20
    NGEN: int = 20
    CXPB: float = 1.0
    MUTPB: float = 0.8
    MAX_CPU_DEMAND: int = 1
    MAX_MEMORY_DEMAND: int = 1
    MIN_AR: float = 0.8

    fileName: str = ""

    if dataType == 0:
        fileName = "latency_lc_ll.csv"
    elif dataType == 1:
        fileName = "latency_lc_hl.csv"
    elif dataType == 2:
        fileName = "latency_hc_ll.csv"
    elif dataType == 3:
        fileName = "latency_hc_hl.csv"

    evolvedPop: "list[creator.Individual]" = evolveInitialWeights(
        POP_SIZE,
        fgs,
        trafficDesign,
        topology,
        MIN_AR,
        MAX_CPU_DEMAND,
        MAX_MEMORY_DEMAND,
        dataType
    )

    TUI.appendToSolverLog("Starting evolution using OpenRASE.")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    evolvedNewPop: "list[creator.Individual]" = []
    for ep in evolvedPop:
        ind = creator.Individual()
        ind.extend(ep)
        evolvedNewPop.append(ind)

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=getWeightLength(fgs, topology),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.8)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = evolvedNewPop
    decodedPop: "list[DecodedIndividual]" = decodePop(pop, topology, fgs)
    gen: int = 1
    for ind in decodedPop:
        ind.fitness.values = evaluate(
            ind,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
            MAX_MEMORY_DEMAND,
            MIN_AR,
            fileName
        )

    hof = tools.ParetoFront()
    hof.update(pop)

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

        decodedOffspring: "list[DecodedIndividual]" = decodePop(offspring, topology, fgs)
        for ind in decodedOffspring:
            ind.fitness.values = evaluate(
                ind,
                gen,
                NGEN,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
                MAX_MEMORY_DEMAND,
                MIN_AR,
                fileName
            )
        pop[:] = toolbox.select(pop + offspring, k=POP_SIZE)

        hof.update(pop)

        gen = gen + 1
