"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from math import floor
import random
from time import sleep
import os
from typing import Callable
from uuid import uuid4, UUID
import pandas as pd
import polars as pl
from deap import base, creator, tools
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.ga_dijkstra_algorithm.ga_utils import decodePop, generateRandomIndividual
from algorithms.models.embedding import DecodedIndividual
from algorithms.surrogacy.utils.extract_weights import getWeightLength
from algorithms.surrogacy.constants.surrogate import SURROGACY_PATH, SURROGATE_DATA_PATH, SURROGATE_PATH
from algorithms.surrogacy.utils.hybrid_evaluation import HybridEvaluation
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
hybridEvolution: "HybridEvaluation" = HybridEvaluation()

def evolveWeights(
    fgs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: TrafficDesign,
    trafficGenerator: TrafficGenerator,
    topology: Topology,
    fileName: str
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
        fileName (str): the name of the file to save the data.

    Returns:
        None
    """

    global isFirstSetWritten

    POP_SIZE: int = 20
    MAX_MEMORY_DEMAND: int = 1

    pop: list[list[list[int]]] = []

    class Individual(list):
        """
        Individual class for DEAP.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.id: UUID = uuid4()

    while len(pop) < POP_SIZE:
        individual = generateRandomIndividual(Individual, topology, fgs, 0.01)
        decodedIndividual: DecodedIndividual = decodePop(
            [individual], topology, fgs
        )[0]
        hybridEvolution.cacheForOnline([decodedIndividual], trafficDesign)
        _maxCPU, maxMemory = hybridEvolution.getMaxCpuMemoryUsageOfHosts(
            decodedIndividual[1], topology, decodedIndividual[2], trafficDesign
        )
        if maxMemory <= MAX_MEMORY_DEMAND and decodedIndividual[4] > 0:
            pop.append(individual)

    decodedPop: list[DecodedIndividual] = decodePop(pop, topology, fgs)
    hybridEvolution.cacheForOffline(
        decodedPop,
        trafficDesign,
        topology,
        0
    )

    for i, ind in enumerate(decodedPop):
        TUI.appendToSolverLog(f"Running iteration {i}.")
        # deploy
        sendEGs(ind[1])

        duration: int = calculateTrafficDuration(trafficDesign[0])
        TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
        TUI.appendToSolverLog(f"Waiting for {duration}s...")

        sleep(duration)

        TUI.appendToSolverLog(f"Done waiting for {duration}s.")

        # process traffic data
        trafficData: pd.DataFrame = trafficGenerator.getData(f"{duration:.0f}s")

        data: pl.DataFrame = hybridEvolution.generateScoresForRealTrafficData(
            ind, trafficData, trafficDesign, topology, i
        )

        with open(
            f"{surrogateDataDirectory}/{fileName}.csv", mode="a", encoding="utf8"
        ) as scoreFile:
            data.write_csv(
                scoreFile,
                include_header=not isFirstSetWritten,
                separator=",",
            )

        isFirstSetWritten = True

        TUI.appendToSolverLog(f"Deleting graphs.")
        deleteEGs(ind[1])
        sleep(30)

    isFirstSetWritten = False
