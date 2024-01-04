"""
Defines models related to telemetry.
"""

from typing import TypedDict


class VNFData(TypedDict):
    """
    Represents the data of a VNF.
    """

    cpuUsage: float
    memoryUsage: float
    networkUsage: float

class SingleHostData(TypedDict):
    """
    Represents the data of a single host.
    """

    cpuUsage: float
    memoryUsage: float
    networkUsage: float
    vnfs: "dict[str, VNFData]"

class HostData(TypedDict):
    """
    Represents the data of the hosts.
    """

    hosts: "dict[str, SingleHostData]"
