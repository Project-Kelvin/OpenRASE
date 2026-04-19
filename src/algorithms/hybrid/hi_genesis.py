"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from typing import Callable
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.utils.hierarchical_evolution import HierarchicalEvolution
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator

NO_OF_NEURONS: int = 2
META_POP_SIZE: int = 10
GENESIS_POP_SIZE: int = 10
META_MAX_GEN: int = 100
GENESIS_MAX_GEN: int = 5
MAX_MEMORY_DEMAND: int = 1
MIN_AR: float = 0.9
MAX_LATENCY: int = 150
MAX_POWER: int = 300
MIN_QUAL_IND: int = 1
META_CXPB: float = 1.0
META_MUTPB: float = 0.5
GENESIS_CXPB: float = 1.0
GENESIS_MUTPB: float = 0.5
META_INDPB: float = 0.5
GENESIS_INDPB: float = 0.5


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

    Returns:
        None
    """

    hiGenesis: HierarchicalEvolution = HierarchicalEvolution(
        META_POP_SIZE,
        GENESIS_POP_SIZE,
        META_CXPB,
        GENESIS_CXPB,
        META_MUTPB,
        GENESIS_MUTPB,
        META_INDPB,
        GENESIS_INDPB,
        META_MAX_GEN,
        GENESIS_MAX_GEN,
        MIN_AR,
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
        retainPopulation,
    )

    hiGenesis.evolve()
