"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import timeit
from typing import Callable, Tuple
from uuid import UUID, uuid4
from deap import base, creator, tools
import numpy as np
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from shared.utils.config import getConfig
from algorithms.models.embedding import DecodedIndividual
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from algorithms.ga_dijkstra_algorithm.ga_utils import (
    convertIndividualToEmbeddingGraph,
    decodePop,
    generateRandomIndividual,
    mutate,
)
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

POP_SIZE: int = 2000

hybridEvolution: HybridEvolution = HybridEvolution(
    "ga_hybrid",
    decodePop,
    generateRandomIndividual,
    tools.cxTwoPoint,
    mutate
)

def solve(
    topology: Topology,
    fgrs: "list[EmbeddingGraph]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: "list[TrafficDesign]",
    trafficGenerator: TrafficGenerator,
    experiment: str
) -> None:
    hybridEvolution.solve(
        topology,
        fgrs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        POP_SIZE,
        experiment
    )
