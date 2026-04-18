"""
Defines the utils for the GENESIS algorithm.
"""

from concurrent.futures import ProcessPoolExecutor
import timeit
from typing import  Optional, Sequence, Tuple, Type, cast
from uuid import uuid4
from deap import tools
import numpy as np
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from algorithms.hybrid.models.individuals import GenesisIndividual
from algorithms.hybrid.models.weights import GenesisWeights
from algorithms.models.embedding import DecodedIndividual, LinkData
from algorithms.hybrid.solvers.chain_composition import generateFGs
from algorithms.hybrid.solvers.link_embedding import EmbedLinks
from algorithms.hybrid.solvers.vnf_embedding import generateEGs
from algorithms.hybrid.utils.extract_weights import (
    generatePredefinedWeights,
    generateRandomWeight,
    getWeights,
    getPredefinedWeights,
    getWeightsLength,
)
from algorithms.hybrid.utils.hybrid_evolution import HybridEvolution, Individual
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI


class GenesisUtils:
    """
    Class responsible for the utils of the GENESIS algorithm.
    """

    predefinedWeights: Optional[GenesisWeights] = None
    noOfNeurons: int = 0
    rejectionRate: float = 0.0
    sigma: float = 0.0

    @classmethod
    def init(
        cls,
        sfcrs: "list[SFCRequest]",
        topology: Topology,
        noOfNeurons: int,
        rejectionRate: float,
        sigma: float,
    ) -> None:
        """
        Sets the predefined weights.

        Parameters:
            sfcrs (list[SFCRequest]): the list of SFCRequests.
            topology (Topology): the topology.
            noOfNeurons (int): the number of neurons in the hidden layer.
            rejectionRate (float): the rejection rate to use for decoding the individuals.
            sigma (float): the sigma to use for decoding the individuals.

        Returns:
            None
        """

        cls.noOfNeurons = noOfNeurons
        cls.predefinedWeights = getPredefinedWeights(
            generatePredefinedWeights(sfcrs, topology, cls.noOfNeurons),
            sfcrs,
            topology,
            cls.noOfNeurons,
        )
        cls.rejectionRate = rejectionRate
        cls.sigma = sigma

    @staticmethod
    def decodeIndividual(
        individual: GenesisIndividual,
        index: int,
        topology: Topology,
        sfcrs: "list[SFCRequest]",
    ) -> DecodedIndividual:
        """
        Decodes an individual to an Embedding Graph.

        Parameters:
            individual (GenesisIndividual): the individual.
            index (int): the index of the individual.
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the list of SFCRequests

        Returns:
            DecodedIndividual: A tuple containing the embedding graphs, embedding data, link data, and acceptance ratio.
        """

        global predefinedWeights

        try:
            weights: GenesisWeights = getWeights(individual, GenesisUtils.noOfNeurons)

            ccPDWeights: "list[float]" = GenesisUtils.predefinedWeights[0] if GenesisUtils.predefinedWeights else []
            vnfPDWeights: "list[float]" = GenesisUtils.predefinedWeights[1] if GenesisUtils.predefinedWeights else []
            linkPDWeights: "list[float]" = GenesisUtils.predefinedWeights[2] if GenesisUtils.predefinedWeights else []
            ccWeights: list[float] = weights[0]
            vnfWeights: list[float] = weights[1]
            linkWeights: list[float] = weights[2]

            fgs: dict[str, list[str]] = generateFGs(
                sfcrs, ccPDWeights, ccWeights, GenesisUtils.noOfNeurons
            )
            egs, nodes, embedData = generateEGs(
                fgs,
                topology,
                vnfPDWeights,
                vnfWeights,
                GenesisUtils.noOfNeurons,
                individual.metaIndividual[0],
                individual.metaIndividual[1],
            )
            embedLinks: Optional[EmbedLinks] = None
            linkData: Optional[LinkData] = None
            if len(egs) > 0:
                embedLinks = EmbedLinks(
                    topology, sfcrs, egs, linkPDWeights, linkWeights, GenesisUtils.noOfNeurons
                )
                egs = embedLinks.embedLinks(nodes)
                linkData = embedLinks.getLinkData()
            ar: float = len(egs) / len(sfcrs)
        except Exception as e:
            TUI.appendToSolverLog(f"Error decoding individual {index}: {e}")
            raise Exception(f"Error decoding individual {index}: {e}")

        return cast(DecodedIndividual, (index, egs, embedData, linkData, ar))

    @staticmethod
    def decodePop(
        pop: list[Individual], topology: Topology, sfcrs: "list[SFCRequest]"
    ) -> list[DecodedIndividual]:
        """
        Generates the Embedding Graphs.

        Parameters:
            pop (list[GenesisIndividual]): the population.
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the list of SFCRequests.

        Returns:
            list[DecodedIndividual]: A list consisting of tuples containing the embedding graphs, embedding data, link data, and acceptance ratio.
        """

        startTime: float = timeit.default_timer()
        decodedPop: "list[DecodedIndividual]" = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    GenesisUtils.decodeIndividual,
                    cast(GenesisIndividual, individual),
                    index,
                    topology,
                    sfcrs,
                )
                for index, individual in enumerate(pop)
            ]

            for future in futures:
                decodedPop.append(future.result())

        endTime: float = timeit.default_timer()
        TUI.appendToSolverLog(
            f"Decoded {len(decodedPop)} individuals in {endTime - startTime:.2f} seconds."
        )

        return decodedPop

    @staticmethod
    def generateRandomGenesisIndividual(
        container: Type[Individual], topology: Topology, sfcrs: "list[SFCRequest]"
    ) -> Individual:
        """
        Generates a random individual.

        Parameters:
            container (Type[Individual]): the container for the individual.
            topology (Topology): the topology.
            sfcrs (list[SFCRequest]): the list of SFCRequests.

        Returns:
            Individual: An individual randomly generated.
        """

        individual: GenesisIndividual = cast(GenesisIndividual, container())
        individual.id = uuid4()
        individual.metaIndividual = Individual([GenesisUtils.rejectionRate, GenesisUtils.sigma])

        weightLength: int = getWeightsLength(GenesisUtils.noOfNeurons)
        for _ in range(weightLength):
            individual.append(generateRandomWeight())

        return individual

    @staticmethod
    def genesisCrossover(
        ind1: Individual,
        ind2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        Crossover between two individuals.

        Parameters:
            ind1 (Individual): the first individual.
            ind2 (Individual): the second individual.

        Returns:
            tuple[Individual, Individual]: the two individuals after crossover.
        """

        return tools.cxBlend(ind1, ind2, alpha=0.5)

    @staticmethod
    def genesisMutate(
        individual: Individual,
        indpb: float,
    ) -> tuple[Individual]:
        """
        Mutates an individual.

        Parameters:
            individual (Individual): the individual to mutate.
            indpb (float): the independent probability for each attribute to be mutated.

        Returns:
            tuple[Individual]: the mutated individual.
        """

        return tools.mutGaussian(individual, mu=0.0, sigma=np.pi, indpb=indpb)
