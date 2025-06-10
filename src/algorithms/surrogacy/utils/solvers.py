""""
Defines the util functions used by the solvers in surrogacy.
"""


from copy import deepcopy
from click import Tuple
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology

from algorithms.models.embedding import DecodedIndividual, EmbeddingData, LinkData
from algorithms.surrogacy.solvers.chain_composition import generateFGs
from algorithms.surrogacy.solvers.link_embedding import EmbedLinks
from algorithms.surrogacy.solvers.vnf_embedding import generateEGs
from algorithms.surrogacy.utils.extract_weights import getWeights


def decodePop(
    pop: "list[list[float]]", sfcrs: "list[SFCRequest]", topology: Topology
) -> list[DecodedIndividual]:
    """
    Generates the Embedding Graphs.

    Parameters:
        individual (list[float]): the individual.
        sfcrs (list[SFCRequest]): the list of SFCRequests.
        topology (Topology): the topology.

    Returns:
        list[DecodedIndividual]: A list consisting of tuples containing the embedding graphs, embedding data, link data, and acceptance ratio.
    """

    decodedPop: "list[DecodedIndividual]" = []

    for i, individual in enumerate(pop):
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

        ar: float = len(egs) / len(sfcrs)

        decodedPop.append(
            (i, egs, embedData, embedLinks.getLinkData(), ar)
        )

    return decodedPop
