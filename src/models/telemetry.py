"""
Defines models related to telemetry.
"""




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

    cpuUsage: float
    memoryUsage: float
    networkUsage: float
    vnfs: "dict[str, VNFData]"


class HostData(dict):
    """
    Represents the data of the hosts.
    """

    hosts: "dict[str, SingleHostData]"


class SrcDstData(dict):
    """
    Represents the data of the source and destination.
    """

    ipSrcDst: str
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
