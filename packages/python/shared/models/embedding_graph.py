"""
Defines the `EmbeddingGraphs` dictionary type and its associated types.
"""

from typing import Optional


class ChainEntity(dict):
    """
    Defines the `ChainEntity` dictionary type.
    """

    id: str
    ip: Optional[str]

class VNFEntity(ChainEntity):
    """
    Defines the `VNFEntity` dictionary type.
    """

    name: Optional[str]

class ForwardingLink:
    """
    Defines the `ForwardingLink` dictionary type.
    """

    source: ChainEntity
    destination: ChainEntity
    links: "list[str]"
    divisor: int


class VNF(dict):
    """
    Defines the `VNF` dictionary type.
    """

    host: ChainEntity
    vnf: VNFEntity
    next: "VNF | list[VNF] | str | list[str]"
    isTraversed: Optional[bool]


class EmbeddingGraph(dict):
    """
    Defines the `EmbeddingGraph` dictionary type.
    """

    sfcID: str
    vnfs: VNF
    links: "list[ForwardingLink]"
    isTraversed: Optional[bool]

# pylint: disable=invalid-name
EmbeddingGraphs = "list[EmbeddingGraph]"
