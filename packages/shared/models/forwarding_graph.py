"""
Defines the `ForwardingGraphs` dictionary type and its associated types.
"""

from typing import Optional, TypedDict


class ChainEntity(TypedDict):
    """
    Defines the `ChainEntity` dictionary type.
    """

    id: str
    ip: Optional[str]


class ForwardingLink:
    """
    Defines the `ForwardingLink` dictionary type.
    """

    source: ChainEntity
    destination: ChainEntity
    links: "list[str]"


class VNF(TypedDict):
    """
    Defines the `VNF` dictionary type.
    """

    host: ChainEntity
    vnf: ChainEntity
    next: "VNF | list[VNF] | str | list[str]"
    isTraversed: Optional[bool]


class ForwardingGraph(TypedDict):
    """
    Defines the `ForwardingGraph` dictionary type.
    """

    sfcID: str
    vnfs: VNF
    links: "list[ForwardingLink]"

# pylint: disable=invalid-name
ForwardingGraphs = "list[ForwardingGraph]"
