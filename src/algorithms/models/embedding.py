"""
This defines the models used for embedding in the algorithms.
"""

from typing import NewType, Tuple
from shared.models.embedding_graph import EmbeddingGraph


EmbeddingData = NewType("EmbeddingData", dict[str, dict[str, list[Tuple[str, int]]]])
LinkData = NewType("LinkData", dict[str, dict[str, tuple[float, float]]])
DecodedIndividual = NewType(
    "DecodedIndividual",
    Tuple[int, list[EmbeddingGraph], EmbeddingData, LinkData, float],
)
