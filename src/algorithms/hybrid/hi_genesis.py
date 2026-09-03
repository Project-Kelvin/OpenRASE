"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from typing import Callable, Union
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.utils.hierarchical_evolution import HierarchicalEvolution
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

NO_OF_NEURONS: int = 2
POP_SIZE: int = 20
MAX_GEN: int = 100
META_POP_SIZE: int = 4
# GENESIS pop size per meta individual
GENESIS_POP_SIZE: int = 5
META_MAX_GEN: int = 20
GENESIS_MAX_GEN: int = 5
MAX_MEMORY_DEMAND: int = 100
MIN_AR: float = 0.95
MAX_LATENCY: int = 100
MAX_POWER: int = 300
MIN_QUAL_IND: int = 1
META_CXPB: float = 1.0
META_MUTPB: float = 0.7
GENESIS_CXPB: float = 1.0
GENESIS_MUTPB: float = 0.7
META_INDPB: float = 0.7
GENESIS_INDPB: float = 0.7
DOMINANCE_THRESHOLD: float = 0.5


def solve(
    sfcrs: "list[SFCRequest]",
    sendEGs: "Callable[[list[EmbeddingGraph]], None]",
    deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
    trafficDesign: list[TrafficDesign],
    trafficGenerator: TrafficGenerator,
    telemetry: Telemetry,
    topology: Topology,
    dirName: str,
    experimentName: str,
    type: str = LATENCY,
    retainPopulation: bool = False,
    isClientMode: bool = False,
    mutPb: float = META_MUTPB,
    indPb: float = META_INDPB,
    cxPb: float = META_CXPB,
    rootIndividual: int = -1,
    minimumAR: float = MIN_AR,
    evaluateOnline: bool = True,
    linesToWrite:list[str] = [],
    dominanceThreshold: float = DOMINANCE_THRESHOLD
) -> None:
    """
    Evolves the weights of the Neural Network.

    Parameters:
        sfcrs (list[SFCRequest]): the list of Service Function Chains.
        sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
        deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
        trafficDesign (list[TrafficDesign]): the traffic design.
        trafficGenerator (TrafficGenerator): the traffic generator.
        telemetry (Telemetry): telemetry instance.
        topology (Topology): the topology.
        dirName (str): the directory name.
        experimentName (str): the name of the experiment.
        type (str): the type of the objective function to optimize. Defaults to LATENCY.
        retainPopulation (bool): specifies if the population should be retained in memory.
        isClientMode (bool): specifies if the algorithm should run in client mode.
        mutPb (float): the mutation probability.
        indPb (float): the individual mutation probability.
        cxPb (float): the crossover probability.
        rootIndividual (int): the index of the root individual in the population. Defaults to -1.
        minimumAR (float): the minimum acceptable arrival rate. Defaults to MIN_AR.
        evaluateOnline (bool): specifies if the evaluation should be done online. Defaults to True.
        linesToWrite (list[str]): the lines to write to the log file. Defaults to [].
        dominanceThreshold (float): the dominance threshold for hyperparameter tuning. Defaults to DOMINANCE_THRESHOLD.

    Returns:
        None
    """

    hiLinesToWrite: list[str] = [
        f"Population Size: {POP_SIZE}",
        f"Max Generations: {MAX_GEN}",
        f"Meta Population Size: {META_POP_SIZE}",
        f"Genesis Population Size: {GENESIS_POP_SIZE}",
        f"Meta Max Generations: {META_MAX_GEN}",
        f"Genesis Max Generations: {GENESIS_MAX_GEN}",
        f"Max Memory Demand: {MAX_MEMORY_DEMAND}",
        f"Min Acceptance Rate: {MIN_AR}",
        f"Max Latency: {MAX_LATENCY}",
        f"Max Power: {MAX_POWER}",
        f"Min Qualification Individual: {MIN_QUAL_IND}",
        f"Meta Crossover Probability: {META_CXPB}",
        f"Meta Individual Mutation Probability: {META_MUTPB}",
        f"Genesis Crossover Probability: {GENESIS_CXPB}",
        f"Genesis Individual Mutation Probability: {GENESIS_MUTPB}",
        f"Meta Gene Mutation Probability: {META_INDPB}",
        f"Genesis Gene Mutation Probability: {GENESIS_INDPB}",
        f"Dominance Threshold: {dominanceThreshold}",
        f"Root Individual: {rootIndividual}",
        f"Evaluate Online: {evaluateOnline}",
    ] + linesToWrite

    hiGenesis: HierarchicalEvolution = HierarchicalEvolution(
        POP_SIZE,
        MAX_GEN,
        cxPb,
        GENESIS_CXPB,
        mutPb,
        GENESIS_MUTPB,
        indPb,
        GENESIS_INDPB,
        minimumAR,
        MAX_LATENCY if type == LATENCY else MAX_POWER,
        NO_OF_NEURONS,
        MAX_MEMORY_DEMAND,
        MIN_QUAL_IND,
        sfcrs,
        topology,
        trafficDesign,
        trafficGenerator,
        telemetry,
        type,
        dirName,
        experimentName,
        sendEGs,
        deleteEGs,
        dominanceThreshold,
        retainPopulation,
        rootIndividual = rootIndividual,
        isClientMode = isClientMode,
        evaluateOnline = evaluateOnline,
        linesToWrite = hiLinesToWrite
    )

    try:
        hiGenesis.evolve()
    except Exception as e:
        TUI.appendToSolverLog(str(e), True)
