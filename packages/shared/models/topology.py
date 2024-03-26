"""
Defines the `Topology` dictionary type and its associated types.
"""




class Host(dict):
    """
    Defines the `Host` dictionary type.
    """

    id: str
    cpu: float
    """
    This is the number of CPUs to use. This is equal to CPU Quota / CPU Period.
    The default CPU Period is 100000. So, in a 4-core machine, a CPU Quota of 200000
    would mean that the host will use 2 cores. You can simply define the number of cores/CPUs to use
    using this `cpu` attribute and the right CPU Quota will be calculated automatically.
    """
    memory: float


class Switch(dict):
    """
    Defines the `Switch` dictionary type.
    """

    id: str


class Link(dict):
    """
    Defines the `Link` dictionary type.
    """

    source: str
    destination: str
    bandwidth: int


class Topology(dict):
    """
    Defines the `Topology` dictionary type.
    """

    hosts: "list[Host]"
    switches: "list[Switch]"
    links: "list[Link]"
