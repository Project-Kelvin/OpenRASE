"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from copy import deepcopy
from multiprocessing import Pool, cpu_count
import random
from time import sleep
from typing import Any, Callable, Tuple
import os
import pandas as pd
import numpy as np
from shared.models.sfc_request import SFCRequest
from algorithms.surrogacy.chain_composition import generateFGs
from deap import base, creator, tools
import tensorflow as tf
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig
from algorithms.surrogacy.extract_weights import getWeightLength, getWeights
from algorithms.surrogacy.local_constants import (
    SURROGACY_PATH,
    SURROGATE_PATH,
)
from algorithms.surrogacy.link_embedding import EmbedLinks
from algorithms.surrogacy.nn import (
    generateEGs,
)
from algorithms.surrogacy.scorer import Scorer
from algorithms.surrogacy.surrogate.surrogate import predict
from models.calibrate import ResourceDemand
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration, getTrafficDesignRate
from utils.tui import TUI

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

directory: str = SURROGACY_PATH
if not os.path.exists(directory):
    os.makedirs(directory)

surrogateDirectory: str = SURROGATE_PATH
if not os.path.exists(surrogateDirectory):
    os.makedirs(surrogateDirectory)

with open(
    f"{SURROGACY_PATH}/data.csv",
    "w",
    encoding="utf8",
) as topologyFile:
    topologyFile.write(
        "generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
    )

with open(
    f"{SURROGACY_PATH}/pfs.csv",
    "w",
    encoding="utf8",
) as pf:
    pf.write("generation, latency, ar\n")


scorer: Scorer = Scorer()

isFirstSetWritten: bool = False


def buildEGs(
    individual: "list[float]", sfcrs: "list[SFCRequest]", topology: Topology
) -> "Tuple[list[EmbeddingGraph],dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]], EmbedLinks]":
    """
    Generates the Embedding Graphs.

    Parameters:
        individual (list[float]): the individual.
        sfcrs (list[SFCRequest]): the list of SFCRequests.
        topology (Topology): the topology.

    Returns:
        Tuple[list[EmbeddingGraph], dict[str, list[str]], dict[str, dict[str, list[Tuple[str, int]]]], EmbedLinks]: (the Embedding Graphs, hosts in the order they should be linked, the embedding data containing the VNFs in hosts).
    """

    weights: "Tuple[list[float], list[float], list[float], list[float]]" = getWeights(
        individual, sfcrs, topology
    )

    ccWeights: "list[float]" = weights[0]
    ccBias: "list[float]" = weights[1]
    vnfWeights: "list[float]" = weights[2]
    vnfBias: "list[float]" = weights[3]
    linkWeights: "list[float]" = weights[4]
    linkBias: "list[float]" = weights[5]

    fgs: "list[EmbeddingGraph]" = generateFGs(sfcrs, ccWeights, ccBias)
    copiedFGs: "list[EmbeddingGraph]" = [deepcopy(fg) for fg in fgs]
    egs, nodes, embedData = generateEGs(copiedFGs, topology, vnfWeights, vnfBias)
    embedLinks: EmbedLinks = None
    if len(egs) > 0:
        embedLinks = EmbedLinks(topology, egs, linkWeights, linkBias)
        egs = embedLinks.embedLinks(nodes)

    return egs, nodes, embedData, embedLinks


def evaluateOnEmulator(
    individualIndex: int,
    individual: "list[float]",
    sfcrs: "list[SFCRequest]",
    gen: int,
    ngen: int,
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    maxMemoryDemand: float,
) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        index (int): individual index.
        individual (list[float]): the individual.
        sfcrs (list[SFCRequest]): the list of Service Function Requests.
        gen (int): the generation.
        ngen (int): the number of generations.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.
        maxMemoryDemand (float): maximum memory demand.

    Returns:
        tuple[float, float]: the fitness.
    """

    global isFirstSetWritten

    egs, _nodes, embedData, embedLinks = buildEGs(individual, sfcrs, topology)

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs) / len(sfcrs)
    latency: int = 0
    penaltyRatio: float = gen / ngen
    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]

    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(sfcrs)} = {acceptanceRatio}"
    )
    if len(egs) > 0:
        sfcIDs: "list[str]" = []
        reqps: "list[float]" = []
        for eg in egs:
            sfcIDs.append(eg["sfcID"])
            reqps.append(maxReqps)

        data: pd.DataFrame = pd.DataFrame(
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
        scorer.cacheData(data, egs)
        scores: "dict[str, ResourceDemand]" = scorer.getHostScores(
            data, topology, embedData
        )
        maxCPU: float = max([score["cpu"] for score in scores.values()])
        maxMemory: float = max([score["memory"] for score in scores.values()])
        TUI.appendToSolverLog(f"Max CPU: {maxCPU}, Max Memory: {maxMemory}")
        # Validate EGs
        # The resource demand of deployed VNFs exceeds the resource capacity of at least 1 host.
        # This leads to servers crashing.
        # Penalty is applied to the latency and the egs are not deployed.
        if maxMemory > maxMemoryDemand:
            TUI.appendToSolverLog(f"Penalty because max Memory demand is {maxMemory}.")
            latency = penaltyLatency * penaltyRatio * (maxMemory)
            acceptanceRatio = acceptanceRatio - (penaltyRatio * (maxMemory))

            return acceptanceRatio, latency

        sendEGs(egs)

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")

        sleep(duration)

        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        trafficData: pd.DataFrame = trafficGenerator.getData(f"{duration + 5:.0f}s")

        if (
            trafficData.empty
            or "_time" not in trafficData.columns
            or "_value" not in trafficData.columns
        ):
            TUI.appendToSolverLog("Traffic data is empty.")
            return 0, penaltyLatency * penaltyRatio

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
            f"{SURROGACY_PATH}/latency.csv",
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
        latency = penaltyLatency * penaltyRatio

    TUI.appendToSolverLog(f"Latency: {latency}ms")

    return acceptanceRatio, latency


def evaluateOnSurrogate(
    individualIndex: int,
    individual: "list[float]",
    sfcrs: "list[SFCRequest]",
    gen: int,
    ngen: int,
    trafficDesign: TrafficDesign,
    topology: Topology,
    maxMemoryDemand: float,
) -> "tuple[float, float]":
    """
    Evaluates the individual.

    Parameters:
        index (int): individual index.
        individual (list[float]): the individual.
        sfcrs (list[SFCRequest]): the list of Service Function Requests.
        gen (int): the generation.
        ngen (int): the number of generations.
        trafficDesign (TrafficDesign): the traffic design.
        topology (Topology): the topology.
        maxMemoryDemand (float): maximum memory demand.

    Returns:
        tuple[float, float]: the fitness.
    """

    egs, _nodes, embedData, embedLinks = buildEGs(individual, sfcrs, topology)

    penaltyLatency: float = 50000
    acceptanceRatio: float = len(egs) / len(sfcrs)
    latency: int = 0
    penaltyRatio: float = gen / ngen
    maxReqps: int = max(trafficDesign[0], key=lambda x: x["target"])["target"]

    TUI.appendToSolverLog(
        f"Acceptance Ratio: {len(egs)}/{len(sfcrs)} = {acceptanceRatio}"
    )

    if len(egs) > 0:
        sfcIDs: "list[str]" = []
        reqps: "list[float]" = []
        for eg in egs:
            sfcIDs.append(eg["sfcID"])
            reqps.append(maxReqps)

        data: pd.DataFrame = pd.DataFrame(
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
        scorer.cacheData(data, egs)
        scores: "dict[str, ResourceDemand]" = scorer.getHostScores(
            data, topology, embedData
        )
        maxCPU: float = max([score["cpu"] for score in scores.values()])
        maxMemory: float = max([score["memory"] for score in scores.values()])
        TUI.appendToSolverLog(f"Max CPU: {maxCPU}, Max Memory: {maxMemory}")
        # Validate EGs
        # The resource demand of deployed VNFs exceeds the resource capacity of at least 1 host.
        # This leads to servers crashing.
        # Penalty is applied to the latency and the egs are not deployed.
        if maxMemory > maxMemoryDemand:
            TUI.appendToSolverLog(f"Penalty because max Memory demand is {maxMemory}.")
            latency = penaltyLatency * penaltyRatio * (maxMemory)
            acceptanceRatio = acceptanceRatio - (penaltyRatio * (maxMemory))

            return acceptanceRatio, latency

        trafficDuration: int = calculateTrafficDuration(trafficDesign[0])
        simulatedReqps: "list[float]" = getTrafficDesignRate(
            trafficDesign[0],
            [1] * trafficDuration,
        )

        simulationData: pd.DataFrame = pd.DataFrame()

        times: "list[int]" = []
        sfcs: "list[str]" = []
        reqpss: "list[float]" = []
        for time, reqps in enumerate(simulatedReqps):
            for eg in egs:
                times.append(time)
                sfcs.append(eg["sfcID"])
                reqpss.append(reqps)

        simulationData = pd.DataFrame(
            {
                "generation": gen,
                "individual": individualIndex,
                "time": times,
                "sfc": sfcs,
                "reqps": reqpss,
                "real_reqps": 0,
                "latency": 0,
                "ar": acceptanceRatio,
            }
        )

        scores: pd.DataFrame = scorer.getSFCScores(
            simulationData, topology, egs, embedData, embedLinks.getLinkData()
        )

        featureDF: pd.DataFrame = scores.groupby(["generation", "individual"]).agg(
            max_cpu=("max_cpu", "mean"),
            link=("link", "mean"),
        )

        prediction: pd.DataFrame = predict(featureDF)

        latency = prediction["PredictedLatency"].mean()

    else:
        latency = penaltyLatency * penaltyRatio

    TUI.appendToSolverLog(f"Surrogate Latency: {latency}ms")

    return acceptanceRatio, latency


def crossover(
    toolbox: base.Toolbox, pop: "list[creator.Individual]", cxpb: float, mutpb: float
) -> "list[creator.Individual]":
    """
    Crossover function.

    Parameters:
        toolbox (base.Toolbox): the toolbox.
        pop (list[creator.Individual]): the population.
        cxpb (float): the crossover probability.
        mutpb (float): the mutation probability.

    Returns:
        list[creator.Individual]: the offspring.
    """

    offspring: "list[creator.Individual]" = list(map(toolbox.clone, pop))
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    return offspring


def select(
    offspring: "list[creator.Individual]",
    pop: "list[creator.Individual]",
    toolbox: base.Toolbox,
    popSize: int,
    hof: tools.ParetoFront,
) -> "Tuple[list[creator.Individual], tools.ParetoFront]":
    """
    Selection function.

    Parameters:
        offspring (list[creator.Individual]): the offspring.
        pop (list[creator.Individual]): the population.
        toolbox (base.Toolbox): the toolbox.
        popSize (int): the population size.
        hof (tools.ParetoFront): the hall of fame.

    Returns:
        Tuple[list[creator.Individual], tools.ParetoFront]: the population and the hall of fame.
    """

    pop[:] = toolbox.select(pop + offspring, k=popSize)

    hof.update(pop)

    return pop, hof


def writeData(gen: int, ars: "list[float]", latencies: "list[float]") -> None:
    """
    Writes the data to the file.

    Parameters:
        gen (int): the generation.
        ars (list[float]): the acceptance ratios.
        latencies (list[float]): the latencies.

    Returns:
        None
    """

    with open(
        f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/data.csv",
        "a",
        encoding="utf8",
    ) as dataFile:
        dataFile.write(
            f"{gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
        )


def writePFs(gen: int, hof: tools.ParetoFront) -> None:
    """
    Writes the Pareto Fronts to the file.

    Parameters:
        gen (int): the generation.
        hof (tools.ParetoFront): the hall of fame.

    Returns:
        None
    """

    TUI.appendToSolverLog(f"Writing Pareto Fronts for generation {gen}.")
    for ind in hof:
        TUI.appendToSolverLog(
            f"{gen}\t {ind.fitness.values[0]}\t {ind.fitness.values[1]}"
        )
        with open(
            f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy/pfs.csv",
            "a",
            encoding="utf8",
        ) as pfFile:
            pfFile.write(f"{gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n")


def evolveWeights(
    sfcrs: "list[SFCRequest]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Service Function Chains.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (TrafficDesign): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        topology (Topology): the topology.

    Returns:
        None
    """

    POP_SIZE: int = 50
    NGEN: int = 100
    CXPB: float = 1.0
    MUTPB: float = 0.8
    MAX_MEMORY_DEMAND: int = 2
    MIN_QUAL_IND: int = 4
    MIN_AR: float = 0.5
    MAX_LATENCY: float = 500

    TUI.appendToSolverLog("Starting the evolution of the weights.")

    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox: base.Toolbox = base.Toolbox()

    toolbox.register("gene", random.uniform, -1, 1)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.gene,
        n=getWeightLength(sfcrs, topology),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("crossover", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=1.0, indpb=0.8)
    toolbox.register("select", tools.selNSGA2)

    pop: "list[creator.Individual]" = toolbox.population(n=POP_SIZE)

    gen: int = 1

    pool: Any = Pool(processes=cpu_count())
    results: tuple[float, float, float] = pool.starmap(
        evaluateOnSurrogate,
        [
            (i, ind, sfcrs, gen, NGEN, trafficDesign, topology, MAX_MEMORY_DEMAND)
            for i, ind in enumerate(pop)
        ],
    )

    for ind, result in zip(pop, results):
        ind.fitness.values = result

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies)

    hof = tools.ParetoFront()
    hof.update(pop)

    writePFs(gen, hof)

    gen = gen + 1

    qualifiedIndividuals: "list[creator.Individual]" = [
        ind
        for ind in hof
        if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
    ]

    while len(qualifiedIndividuals) < MIN_QUAL_IND:
        offspring: "list[creator.Individual]" = crossover(toolbox, pop, CXPB, MUTPB)

        results: tuple[float, float, float] = pool.starmap(
            evaluateOnSurrogate,
            [
                (i, ind, sfcrs, gen, NGEN, trafficDesign, topology, MAX_MEMORY_DEMAND)
                for i, ind in enumerate(offspring)
            ],
        )

        for ind, result in zip(offspring, results):
            ind.fitness.values = result

        pop, hof = select(offspring, pop, toolbox, POP_SIZE, hof)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies)
        writePFs(gen, hof)

        qualifiedIndividuals = [
            ind
            for ind in hof
            if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
        ]

        minAR = min(ars)
        maxLatency = max(latencies)

        TUI.appendToSolverLog(
            f"Generation {gen}: Min AR: {minAR}, Max Latency: {maxLatency}"
        )

        TUI.appendToSolverLog(
            f"Qualified Individuals: {len(qualifiedIndividuals)}/{MIN_QUAL_IND}"
        )

        gen = gen + 1

    TUI.appendToSolverLog(
        f"Finished the evolution of weights using surrogate at generation {gen - 1}."
    )
    TUI.appendToSolverLog(
        f"Number of qualified individuals: {len(qualifiedIndividuals)}"
    )

    pop = qualifiedIndividuals
    emHof = tools.ParetoFront()
    emPopSize = len(pop)
    emQualifiedIndividuals: "list[creator.Individual]" = []
    EM_MIN_QUAL_IND: int = 1

    for i, ind in enumerate(pop):
        ind.fitness.values = evaluateOnEmulator(
            i,
            ind,
            sfcrs,
            gen,
            NGEN,
            sendEGs,
            deleteEGs,
            trafficDesign,
            trafficGenerator,
            topology,
            MAX_MEMORY_DEMAND,
        )

    emHof.update(pop)

    ars = [ind.fitness.values[0] for ind in pop]
    latencies = [ind.fitness.values[1] for ind in pop]

    writeData(gen, ars, latencies)
    writePFs(gen, emHof)

    emQualifiedIndividuals = [
        ind
        for ind in emHof
        if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
    ]

    emMinAR = min(ars)
    emMaxLatency = max(latencies)

    TUI.appendToSolverLog(
        f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
    )

    gen = gen + 1

    while len(emQualifiedIndividuals) < EM_MIN_QUAL_IND:
        offspring: "list[creator.Individual]" = crossover(toolbox, pop, CXPB, MUTPB)
        for i, ind in enumerate(offspring):
            ind.fitness.values = evaluateOnEmulator(
                i,
                ind,
                sfcrs,
                gen,
                NGEN,
                sendEGs,
                deleteEGs,
                trafficDesign,
                trafficGenerator,
                topology,
                MAX_MEMORY_DEMAND,
            )

        pop, emHof = select(offspring, pop, toolbox, emPopSize, emHof)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        writeData(gen, ars, latencies)
        writePFs(gen, emHof)

        emQualifiedIndividuals = [
            ind
            for ind in emHof
            if ind.fitness.values[0] >= MIN_AR and ind.fitness.values[1] <= MAX_LATENCY
        ]

        emMinAR = min(ars)
        emMaxLatency = max(latencies)

        TUI.appendToSolverLog(
            f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
        )

        gen = gen + 1
