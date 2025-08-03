"""
This defines the GAHA algorithm of Mohammad Ali Khoshkholghi.
"""

import copy
from time import sleep
import timeit
from typing import Callable
from deap import base, creator, tools
import pandas as pd
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from algorithms.mak_ga.utils import (
    cacheDemand,
    convertIndividualToEmbeddingGraphs,
    decodePop,
    generateRandomIndividual,
    getTotalDelay,
    isHostConstraintViolated,
    isLinkConstraintViolated,
)
from algorithms.models.embedding import EmbeddingData, LinkData
from algorithms.surrogacy.utils.demand_predictions import DemandPredictions
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import calculateTrafficDuration


def gahaEvolve(
    fgrs: list[EmbeddingGraph],
    topology: Topology,
    trafficDesign: list[TrafficDesign],
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficGenerator: TrafficGenerator,
) -> None:
    """
    This function starts the GAHA.
    """

    startTime: float = timeit.default_timer()

    ALPHA: float = 0.3
    BETA: int = 100
    ELITISM_RATE: float = 0.2
    POP_SIZE: int = 100
    NGEN: int = 100

    noOfHosts: int = len(topology["hosts"])
    demandPredictions: DemandPredictions = DemandPredictions()

    selectionRate: int = round(POP_SIZE * (1 - ELITISM_RATE))
    elitismRate: int = POP_SIZE - selectionRate

    creator.create("MaxARMinLatency", base.Fitness, weights=(-1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox: base.Toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        generateRandomIndividual,
        creator.Individual,
        fgrs,
        trafficDesign,
        topology,
        demandPredictions,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt, 0, noOfHosts, indpb=1.0)
    toolbox.register("select", tools.selRoulette)

    pop = toolbox.population(n=POP_SIZE)
    gen: int = 1

    while gen <= NGEN:
        decodedPop: list[
            tuple[
                int,
                list[EmbeddingGraph],
                EmbeddingData,
                LinkData,
                float,
                dict[str, list[tuple[str, int]]],
            ]
        ] = decodePop(pop, fgrs, topology)

        cacheDemand(decodedPop, demandPredictions, trafficDesign)

        delays: list[float] = []
        ars: list[float] = []
        for ind in decodedPop:
            delay: float = getTotalDelay(
                ind[1], topology, demandPredictions, trafficDesign
            )
            ar: float = (1 / ind[4]) if ind[4] != 0 else 10000
            delays.append(delay)
            ars.append(ar)
        for ind, delay, ar in zip(pop, delays, ars):
            delay = (
                (delay - min(delays)) / (max(delays) - min(delays))
                if max(delays) != min(delays)
                else 0
            )
            ar = (ar - min(ars)) / (max(ars) - min(ars)) if max(ars) != min(ars) else 0
            ind.fitness.values = (ALPHA * ar + (1 - ALPHA) * delay,)

        sortedPop: list[creator.Individual] = pop.sort(
            key=lambda x: x.fitness.values[0], reverse=False
        )
        newPop: list[creator.Individual] = sortedPop[:elitismRate]

        offspring: list[creator.Individual] = []

        while len(offspring) < selectionRate:
            parents: list[creator.Individual] = toolbox.select(sortedPop, k=2)

            isValid: bool = False
            attempts: int = 0

            while not isValid and attempts < BETA:
                child1: creator.Individual = copy.deepcopy(parents[0])
                child2: creator.Individual = copy.deepcopy(parents[1])

                toolbox.mate(child1, child2)

                decodedChildren: list[
                    tuple[
                        int,
                        list[EmbeddingGraph],
                        EmbeddingData,
                        LinkData,
                        float,
                        dict[str, list[tuple[str, int]]],
                    ]
                ] = decodePop([child1, child2], fgrs, topology)
                cacheDemand(decodedChildren, demandPredictions, trafficDesign)
                child1HostConstraintViolated: bool = isHostConstraintViolated(
                    decodedChildren[0], topology, demandPredictions, trafficDesign
                )
                child2HostConstraintViolated: bool = isHostConstraintViolated(
                    decodedChildren[1], topology, demandPredictions, trafficDesign
                )
                child1LinkConstraintViolated: bool = isLinkConstraintViolated(
                    decodedChildren[0], topology, trafficDesign
                )
                child2LinkConstraintViolated: bool = isLinkConstraintViolated(
                    decodedChildren[1], topology, trafficDesign
                )

                isValid = (
                    not child1HostConstraintViolated
                    and not child2HostConstraintViolated
                    and not child1LinkConstraintViolated
                    and not child2LinkConstraintViolated
                )
                attempts += 1

            attempts = 0
            isValid = False
            while not isValid and attempts < BETA:
                child1: creator.Individual = copy.deepcopy(parents[0])
                child2: creator.Individual = copy.deepcopy(parents[1])

                toolbox.mutate(child1)
                toolbox.mutate(child2)

                decodedChildren: list[
                    tuple[
                        int,
                        list[EmbeddingGraph],
                        EmbeddingData,
                        LinkData,
                        float,
                        dict[str, list[tuple[str, int]]],
                    ]
                ] = decodePop([child1, child2], fgrs, topology)
                cacheDemand(decodedChildren, demandPredictions, trafficDesign)
                child1HostConstraintViolated: bool = isHostConstraintViolated(
                    decodedChildren[0], topology, demandPredictions, trafficDesign
                )
                child2HostConstraintViolated: bool = isHostConstraintViolated(
                    decodedChildren[1], topology, demandPredictions, trafficDesign
                )
                child1LinkConstraintViolated: bool = isLinkConstraintViolated(
                    decodedChildren[0], topology, trafficDesign
                )
                child2LinkConstraintViolated: bool = isLinkConstraintViolated(
                    decodedChildren[1], topology, trafficDesign
                )

                isValid = (
                    not child1HostConstraintViolated
                    and not child2HostConstraintViolated
                    and not child1LinkConstraintViolated
                    and not child2LinkConstraintViolated
                )
                attempts += 1

            offspring.append(child1)
            offspring.append(child2)

        offspring = offspring[:selectionRate]

        newPop.extend(offspring)
        pop[:] = newPop
        gen += 1

    endTime: float = timeit.default_timer()

    print(f"GAHA finished in {endTime - startTime:.2f} seconds.")

    bestIndividual: creator.Individual = tools.selBest(pop, k=1)[0]
    decodedBestInd: tuple[
        list[EmbeddingGraph], EmbeddingData, LinkData, dict[str, list[tuple[str, int]]]
    ] = convertIndividualToEmbeddingGraphs(bestIndividual, fgrs, topology)

    print(f"Best Individual Fitness: {bestIndividual.fitness.values[0]}")
    print(f"Acceptance Ratio: {decodedBestInd[4]}")

    sendEGs(decodedBestInd[0])
    duration: int = calculateTrafficDuration(trafficDesign)
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

    print(f"Average Latency: {latency:.2f} ms")
    deleteEGs(decodedBestInd[0])
