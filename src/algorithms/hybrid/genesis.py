"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from typing import Callable
import tensorflow as tf
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.models.individuals import GenesisIndividual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

NO_OF_NEURONS: int = 2
POP_SIZE: int = 100
REJECTION_RATE: float = 0.05
SIGMA: float = 2.0


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

    GenesisUtils.init(sfcrs, topology, NO_OF_NEURONS, REJECTION_RATE, SIGMA)

    hybridEvolution: HybridEvolution = HybridEvolution(
        dirName,
        GenesisUtils.decodePop,
        GenesisUtils.generateRandomGenesisIndividual,
        GenesisUtils.genesisCrossover,
        GenesisUtils.genesisMutate,
        GenesisIndividual,
    )

    hybridEvolution.hybridSolve(
        topology,
        sfcrs,
        sendEGs,
        deleteEGs,
        trafficDesign,
        trafficGenerator,
        telemetry,
        POP_SIZE,
        experimentName,
        type,
        retainPopulation,
    )
