"""
This defines the GA that evolves teh weights of the Neural Network.
"""

from typing import Callable, Type
import numpy as np
import tensorflow as tf
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from algorithms.hybrid.constants.genesis_objective import LATENCY
from algorithms.hybrid.models.individuals import GenesisIndividual, Individual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution
from algorithms.models.embedding import DecodedIndividual
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator

tf.get_logger().setLevel("ERROR")
tf.keras.utils.disable_interactive_logging()

NO_OF_NEURONS: int = 2
POP_SIZE: int = 20
REJECTION_RATE: float = 0.05
SIGMA: float = 2.0
MUTPB: float = 0.7
INDPB: float = 0.7
CXPB: float = 1.0
ACTIVATION: str = "sin"
INIT_LIMIT: float = 2 * np.pi


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
    mutPb: float = MUTPB,
    indPb: float = INDPB,
    cxPb: float = CXPB,
    sigma: float = SIGMA,
    rejectionRate: float = REJECTION_RATE,
    evaluateOnline: bool = True,
    staticChain: bool = False,
    dijkstra: bool = False,
    disableGaussian: bool = False,
    activation: str = ACTIVATION,
    initLimit: float = INIT_LIMIT
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
        mutPb (float): the mutation probability.
        indPb (float): the individual mutation probability.
        cxPb (float): the crossover probability.
        sigma (float): the standard deviation for the Gaussian noise.
        rejectionRate (float): the rate at which individuals are rejected.
        evaluateOnline (bool): whether to evaluate the solution online or offline.
        staticChain (bool): whether to use static chain decoding.
        dijkstra (bool): whether to use Dijkstra's algorithm for pathfinding.
        disableGaussian (bool): whether to disable the Gaussian distribution for host selection.
        activation (str): the type of activation function to apply.
        initLimit (float): the limit to use for generating the predefined weights.

    Returns:
        None
    """

    GenesisUtils.init(sfcrs, topology, NO_OF_NEURONS, rejectionRate, sigma)

    def decodePopWrapper(
        pop: list[Individual],
        topology: Topology,
        sfcrs: "list[SFCRequest]"
    ) -> list[DecodedIndividual]:
        """
        A thunk function to decode the population.

        Parameters:
            pop (list[Individual]): the population.
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the list of Service Function Chains.

        Returns:
            list[GenesisUtils.DecodedIndividual]: the decoded population.
        """

        return GenesisUtils.decodePop(pop, topology, sfcrs, staticChain, dijkstra, disableGaussian, activation)

    def mutateWrapper(
        individual: Individual,
        indpb: float
    ) -> Individual:
        """
        A thunk function to mutate an individual.

        Parameters:
            individual (Individual): the individual to mutate.
            indpb (float): the individual mutation probability.

        Returns:
            Individual: the mutated individual.
        """

        return GenesisUtils.genesisMutate(individual, indpb, initLimit=initLimit)

    def generateRandomIndividualWrapper(container: Type[Individual], topology: Topology, sfcrs: "list[SFCRequest]") -> Individual:
        """
        A thunk function to generate a random individual.

        Parameters:
            container (Type[Individual]): the type of individual to generate.
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the list of Service Function Chains.

        Returns:
            Individual: the generated individual.
        """

        return GenesisUtils.generateRandomGenesisIndividual(container, topology, sfcrs, initLimit=initLimit)

    linesToWrite: list[str] = [
        f"Is VNF CC Disabled: {staticChain}",
        f"Is Dijkstra Used: {dijkstra}",
        f"Is Gaussian Disabled: {disableGaussian}",
        f"Activation Function: {activation}",
        f"Initial Weight Limit: {initLimit}"
    ]

    hybridEvolution: HybridEvolution = HybridEvolution(
        dirName,
        decodePopWrapper,
        generateRandomIndividualWrapper,
        GenesisUtils.genesisCrossover,
        mutateWrapper,
        GenesisIndividual,
        mutPb,
        cxPb,
        indPb,
        evaluateOnline=evaluateOnline
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
        linesToWrite=linesToWrite
    )
