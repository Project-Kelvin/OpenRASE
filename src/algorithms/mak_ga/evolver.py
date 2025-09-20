"""
This defines the GAHA algorithm of Mohammad Ali Khoshkholghi.
"""

from concurrent.futures import ProcessPoolExecutor
import copy
import os
import random
from time import sleep
import timeit
from typing import Callable
from deap import base, creator, tools
import numpy as np
import pandas as pd
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from algorithms.mak_ga.mak_ga_utils import MakGAUtils
from algorithms.models.embedding import EmbeddingData, LinkData
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI

artifactsDir: str = os.path.join(getConfig()["repoAbsolutePath"], "artifacts", "experiments", "gaha")

if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)


def gahaEvolve(
    fgrs: list[EmbeddingGraph],
    topology: Topology,
    trafficDesign: list[TrafficDesign],
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficGenerator: TrafficGenerator,
    experiment: str
) -> None:
    """
    This function starts the GAHA.
    """

    experimentDir: str = os.path.join(artifactsDir, experiment)

    if not os.path.exists(experimentDir):
        os.makedirs(experimentDir)

    with open(os.path.join(experimentDir, f"data.csv"), "w") as expFile:
        expFile.write("generation,avg_ar,max_ar,min_ar,avg_latency,max_latency,min_latency\n")

    startTime: float = timeit.default_timer()

    ALPHA: float = 0.3
    BETA: int = 100
    ELITISM_RATE: float = 0.2
    POP_SIZE: int = 100
    NGEN: int = 500
    MIN_AR: float = 1.0
    MAX_LATENCY: float = 500.0

    noOfHosts: int = len(topology["hosts"])
    makGAUtils: MakGAUtils = MakGAUtils(topology, trafficDesign[0], fgrs)

    selectionRate: int = round(POP_SIZE * (1 - ELITISM_RATE))
    elitismRate: int = POP_SIZE - selectionRate

    creator.create("MaxARMinLatency", base.Fitness, weights=(-1.0,))
    creator.create("prob", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency, prob=creator.prob)

    toolbox: base.Toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        makGAUtils.generateRandomIndividual,
        creator.Individual
    )

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    pop = toolbox.population(n=POP_SIZE)
    TUI.appendToSolverLog("Initial population created.")
    gen: int = 1
    validIndividual: tuple[float, float, list[EmbeddingGraph], float] = None
    delays: list[float] = []
    ars: list[float] = []
    realARs: list[float] = []

    while validIndividual is None:
        TUI.appendToSolverLog(f"Generation {gen} started.")
        decodedPop: list[
            tuple[
                int,
                list[EmbeddingGraph],
                EmbeddingData,
                LinkData,
                float,
                dict[str, list[tuple[str, int]]],
            ]
        ] = makGAUtils.decodePop(pop)
        makGAUtils.cacheDemand(decodedPop)

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    makGAUtils.getTotalDelay,
                    ind,
                )
                for ind in decodedPop
            ]
            for i, future in enumerate(futures):
                delay = future.result()
                ind = decodedPop[i]
                ar: float = (1 / ind[4]) if ind[4] != 0 else 10000
                delays.append(delay)
                ars.append(ar)
                realARs.append(ind[4])

        with open(os.path.join(experimentDir, f"data.csv"), "a") as expFile:
            expFile.write(
                f"{gen},{np.mean(realARs)},{max(realARs)},{min(realARs)},{np.mean(delays)},{max(delays)},{min(delays)}\n"
            )

        for ind, delay, ar, decodedInd in zip(pop, delays, ars, decodedPop):
            delay = (
                ((delay - min(delays)) / (max(delays) - min(delays)))
                if max(delays) != min(delays)
                else 0
            )
            ar = ((ar - min(ars)) / (max(ars) - min(ars))) if max(ars) != min(ars) else 0
            ind.fitness.values = (ALPHA * ar + (1 - ALPHA) * delay,)
            if decodedInd[4] >= MIN_AR and delay <= MAX_LATENCY:
                validIndividual = (decodedInd[4], delay, decodedInd[1], ind.fitness.values[0])

                break

        if gen >= NGEN:
            break

        pop.sort(
            key=lambda x: x.fitness.values[0], reverse=False
        )

        newPop: list[creator.Individual] = pop[:elitismRate]

        offspring: list[creator.Individual] = []

        for ind in pop:
            fitnessSum: float = sum(ind1.fitness.values[0] for ind1 in pop)
            ind.prob.values = ((ind.fitness.values[0] / fitnessSum) if fitnessSum != 0 else 0,)

        minProb: float = min(ind.prob.values[0] for ind in pop)
        maxProb: float = max(ind.prob.values[0] for ind in pop)

        maxMinDiff: float = maxProb - minProb
        for ind in pop:
            ind.prob.values = ((((1 - 10) * ((ind.prob.values[0] - minProb) / maxMinDiff) + 10) if maxMinDiff != 0 else 0),)

        while len(offspring) < selectionRate:
            parents: list[creator.Individual] = tools.selRoulette(pop, k=2, fit_attr="prob")
            isValid: bool = False
            attempts: int = 0
            selectedChild: creator.Individual = None

            while not isValid and attempts < BETA:
                child1: creator.Individual = copy.deepcopy(parents[0])
                child2: creator.Individual = copy.deepcopy(parents[1])

                newChild: creator.Individual = creator.Individual([0] * len(child1))
                for gene in range(len(child1)):
                    r: int = random.choice([0,1])
                    if r <= child2.fitness.values[0] / (child1.fitness.values[0] + child2.fitness.values[0]):
                        newChild[gene] = child2[gene]
                    else:
                        newChild[gene] = child1[gene]

                decodedChildren: list[
                    tuple[
                        int,
                        list[EmbeddingGraph],
                        EmbeddingData,
                        LinkData,
                        float,
                        dict[str, list[tuple[str, int]]],
                    ]
                ] = makGAUtils.decodePop([newChild])
                makGAUtils.cacheDemand(decodedChildren)
                # child1HostConstraintViolated: bool = isHostConstraintViolated(
                #     decodedChildren[0], topology, demandPredictions, trafficDesign[0]
                # )
                # child1LinkConstraintViolated: bool = isLinkConstraintViolated(
                #     decodedChildren[0], topology, trafficDesign[0]
                # )

                # isValid = (
                #     not child1HostConstraintViolated
                #     and not child1LinkConstraintViolated
                # )
                isValid = True

                if isValid:
                    selectedChild = copy.deepcopy(newChild)

                attempts += 1

            if selectedChild is None:
                if parents[0].fitness.values[0] > parents[1].fitness.values[0]:
                    selectedChild = copy.deepcopy(parents[0])
                else:
                    selectedChild = copy.deepcopy(parents[1])
            offspring.append(selectedChild)

        newPop = newPop + offspring
        mutatedPop: list[creator.Individual] = []
        for pop in newPop:
            attempts = 0
            isValid = False
            selectedChild: creator.Individual = None
            while not isValid and attempts < BETA:
                child1: creator.Individual = copy.deepcopy(pop)

                gene: int = random.randint(0, len(child1) - 1)
                host: int = random.randint(0, noOfHosts - 1)

                child1[gene] = host

                decodedChildren: list[
                    tuple[
                        int,
                        list[EmbeddingGraph],
                        EmbeddingData,
                        LinkData,
                        float,
                        dict[str, list[tuple[str, int]]],
                    ]
                ] = makGAUtils.decodePop([child1])
                makGAUtils.cacheDemand(decodedChildren)
                # child1HostConstraintViolated: bool = isHostConstraintViolated(
                #     decodedChildren[0], topology, demandPredictions, trafficDesign[0]
                # )
                # child1LinkConstraintViolated: bool = isLinkConstraintViolated(
                #     decodedChildren[0], topology, trafficDesign[0]
                # )

                # isValid = (
                #     not child1HostConstraintViolated
                #     and not child1LinkConstraintViolated
                # )
                isValid = True

                if isValid:
                    selectedChild = copy.deepcopy(child1)

                attempts += 1

            if selectedChild is None:
                selectedChild = copy.deepcopy(pop)

            mutatedPop.append(selectedChild)

        pop[:] = mutatedPop
        gen += 1
        delays = []
        ars = []
        realARs = []

    endTime: float = timeit.default_timer()

    egToTest: list[EmbeddingGraph] = None
    bestAR: float = -1
    bestCompLatency: float = -1
    bestFitness: float = -1

    if validIndividual is not None:
        TUI.appendToSolverLog(f"Valid individual found in generation {gen}.")
        egToTest = validIndividual[2]
        bestAR = validIndividual[0]
        bestCompLatency = validIndividual[1]
        bestFitness = validIndividual[3]
    else:
        TUI.appendToSolverLog(f"No valid individual found in {NGEN} generations.")
        maxFitness: float = -1
        for ind, decodedInd, realAR, delay in zip(pop, decodedPop, realARs, delays):
            if ind.fitness.values[0] > maxFitness:
                maxFitness = ind.fitness.values[0]
                egToTest = decodedInd[1]
                bestAR = realAR
                bestCompLatency = delay
                bestFitness = ind.fitness.values[0]

    sendEGs(egToTest)
    duration: int = calculateTrafficDuration(trafficDesign[0])
    sleep(duration)

    trafficData: pd.DataFrame = trafficGenerator.getData(duration)

    trafficData["_time"] = trafficData["_time"] // 1000000000

    groupedTrafficData: pd.DataFrame = trafficData.groupby(
        ["_time", "sfcID"]
    ).agg(
        reqps=("_value", "count"),
        medianLatency=("_value", "median"),
    )

    latency: float = groupedTrafficData["medianLatency"].mean()

    deleteEGs(egToTest)

    names: list[str] = experiment.split("_")
    with open(
        os.path.join(experimentDir, f"experiment.txt"),
        "w",
        encoding="utf8",
    ) as expFile:
        expFile.write(f"No. of SFCRs: {4 * int(names[0])}\n")
        expFile.write(f"Traffic Scale: {float(names[1]) * 10}\n")
        expFile.write(
            f"Traffic Pattern: {'Pattern B' if names[2] == 'True' else 'Pattern A'}\n"
        )
        expFile.write(f"Link Bandwidth: {names[3]}\n")
        expFile.write(f"No. of CPUs: {names[4]}\n")
        expFile.write(f"Evolution Time taken: {endTime - startTime:.2f}\n")
        expFile.write(f"Experiment Time: {duration}s\n")

        expFile.write(f"Acceptance Ratio: {bestAR}\n")
        expFile.write(f"Average Latency(computed): {bestCompLatency}\n")
        expFile.write(f"Fitness score: {bestFitness}\n")
        expFile.write(f"Average Latency(measured): {latency}\n")
