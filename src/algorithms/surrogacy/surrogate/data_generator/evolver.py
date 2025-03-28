"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
from math import floor
import random
from time import sleep
from timeit import default_timer
from typing import Callable, Tuple
import os
import pandas as pd
from deap import base, creator, tools
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.surrogacy.extract_weights import getWeightLength
from algorithms.surrogacy.local_constants import SURROGACY_PATH, SURROGATE_DATA_PATH, SURROGATE_PATH
from algorithms.surrogacy.surrogate.data_generator.init_pop_generator import (
    evolveInitialWeights,
    getWeights,
)
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.nn import convertDFtoEGs, convertFGsToDF, getConfidenceValues
from algorithms.surrogacy.scorer import Scorer
from models.calibrate import ResourceDemand
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration, getTrafficDesignRate
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

scorer: Scorer = Scorer()

isFirstSetWritten: bool = False

def evaluate(
    individualIndex: int,
    individual: "list[float]",
    fgs: "list[EmbeddingGraph]",
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
        maxMemoryDemand (float): maximum memory demand.
        minAR (float): minimum acceptance ratio.
        fileName (str): the file name.

    Returns:
        tuple[float, float]: the fitness.
    """

    global isFirstSetWritten

    # Decode individual
    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(
        individual, copiedFGs, topology
    )
    df: pd.DataFrame = convertFGsToDF(copiedFGs, topology)
    newDF: pd.DataFrame = getConfidenceValues(df, weights[0], weights[1])
    egs, nodes, embedData = convertDFtoEGs(newDF, copiedFGs, topology)
    if len(egs) > 0:
        embedLinks: EmbedLinks = EmbedLinks(topology, egs, weights[2], weights[3])
        start: float = default_timer()
        egs = embedLinks.embedLinks(nodes)
        end: float = default_timer()
        TUI.appendToSolverLog(f"Link Embedding Time for all EGs: {end - start}s")

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs) / len(fgs)
    latency: int = 0
    penaltyWeight: float = gen / ngen
    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]

    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(fgs)} = {acceptanceRatio}"
    )

    if len(egs) > 0:
        # Check individual validity
        sfcIDs: "list[str]" = []
        reqps: "list[float]" = []
        for eg in egs:
            sfcIDs.append(eg["sfcID"])
            reqps.append(maxReqps)

        hostData: pd.DataFrame = pd.DataFrame(
            {
                "generation": 0,
                "individual": 0,
                "time": 0,
                "sfc": sfcIDs,
                "reqps": reqps,
                "real_reqps": 0,
                "latency": 0,
                "ar": acceptanceRatio,
            }
        )
        scorer.cacheData(hostData, egs)
        scores: "dict[str, ResourceDemand]" = scorer.getHostScores(
            hostData, topology, embedData
        )
        maxMemory: float = max([score["memory"] for score in scores.values()])
        TUI.appendToSolverLog(f"Max Memory: {maxMemory}")

        # Validate EGs
        # The resource demand of deployed VNFs exceeds the resource capacity of at least 1 host.
        # This leads to servers crashing.
        # Penalty is applied to the latency and the egs are not deployed.
        if maxMemory > maxMemoryDemand:
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

        trafficData["_time"] = trafficData["_time"] // 1000000000

        groupedTrafficData: pd.DataFrame = trafficData.groupby(["_time", "sfcID"]).agg(
            reqps=("_value", "count"),
            medianLatency=("_value", "median"),
        )

        simulatedReqps: "list[float]" = getTrafficDesignRate(
            trafficDesign[0],
            [1] * groupedTrafficData.index.get_level_values(0).unique().size,
        )

        latency: float = 0

        index: int = 0
        time: "list[int]" = []
        sfcIDs: "list[str]" = []
        reqps: "list[float]" = []
        realReqps: "list[float]" = []
        latencies: "list[float]" = []
        ars: "list[float]" = []
        generation: "list[int]" = []
        for i, group in groupedTrafficData.groupby(level=0):
            for eg in egs:
                generation.append(gen)
                time.append(i)
                sfcIDs.append(eg["sfcID"])
                reqps.append(
                    simulatedReqps[index]
                    if index < len(simulatedReqps)
                    else simulatedReqps[-1]
                )
                realReqps.append(
                    group.loc[(i, eg["sfcID"])]["reqps"]
                    if eg["sfcID"] in group.index.get_level_values(1)
                    else 0
                )
                latencies.append(
                    group.loc[(i, eg["sfcID"])]["medianLatency"]
                    if eg["sfcID"] in group.index.get_level_values(1)
                    else 0
                )
                ars.append(acceptanceRatio)
            index += 1
        data: pd.DataFrame = pd.DataFrame(
            {
                "generation": generation,
                "individual": individualIndex,
                "time": time,
                "sfc": sfcIDs,
                "reqps": reqps,
                "real_reqps": realReqps,
                "latency": latencies,
                "ar": ars,
            }
        )

        data = scorer.getSFCScores(
            data, topology, egs, embedData, embedLinks.getLinkData()
        )

        data.to_csv(
            f"{surrogateDataDirectory}/{fileName}",
            mode="a" if isFirstSetWritten else "w",
            header=not isFirstSetWritten,
            index=False,
            encoding="utf8",
        )

        isFirstSetWritten = True

        data["latency"] = data["latency"].replace(0, 1500)

        latency = data["latency"].mean()

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

    gen: int = 1
    for i, ind in enumerate(pop):
        ind.fitness.values = evaluate(
            i,
            ind,
            fgs,
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

        for i, ind in enumerate(offspring):
            ind.fitness.values = evaluate(
                i,
                ind,
                fgs,
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
