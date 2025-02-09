"""
Defines models related to telemetry.
"""




from typing import Union


class VNFData(dict):
    """
    Represents the data of a VNF.
    """

    cpuUsage: float
    memoryUsage: float
    networkUsage: float


class SingleHostData(dict):
    """
    Represents the data of a single host.
    """

    cpuUsage: "tuple[float, float, float]"
    memoryUsage: "tuple[float, float, float]"
    networkUsage: float
    vnfs: "dict[str, VNFData]"


class HostData(dict):
    """
    Represents the data of the hosts.
    """

    startTime: int
    endTime: int
    hostData: "dict[str, SingleHostData]"

class SrcDstData(dict):
    """
    Represents the data of the source and destination.
    """

    ipSrcDst: "tuple[tuple[str, str], tuple[str, str]]"
    interface: str
    value: float


class FlowData(dict):
    """
    Represents the data of the flows.
    """

    interface: str
    value: float


class SwitchData(dict):
    """
    Represents the data of the switches.
    """

    ipSrcDst: "list[SrcDstData]"
    inflow: "list[FlowData]"
    outflow: "list[FlowData]"
    timeStamp: int
