"""
Defines the model related GA individuals and weights.
"""

from typing import NewType


GenesisWeights = NewType(
    "GenesisWeights",
    tuple[list[float], list[float], list[float]],
)
