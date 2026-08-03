"""
Defines the models used in the GAHA algorithm.
"""

from typing import NewType


EmbeddingData = NewType("EmbeddingData", dict[str, dict[str, list[tuple[str, int, int]]]])
