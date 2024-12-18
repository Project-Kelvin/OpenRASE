"""
This defines the logic to add host to the Mininet environment.
"""

from typing import Any
from shared.models.topology import Host
from mininet.node import Host as MininetHost
from shared.utils.config import getConfig

from constants.container import CPU_PERIOD, DIND_IMAGE

def addHostNode(host: Host, net: Any) -> MininetHost:
    """
    Add a host node to the Mininet environment.

    Parameters:
        host (Host): The host to add.
        net (Any): The Mininet environment.

    Returns:
        MininetHost: The host node.
    """

    return net.addDocker(
        f"{host['id']}Node",
        ip=f"{getConfig()['sff']['network1']['hostIP']}/{getConfig()['sff']['network1']['mask']}",
        cpu_quota=int(host["cpu"] * CPU_PERIOD if "cpu" in host else -1),
        mem_limit=(
            f"{host['memory']}mb"
            if "memory" in host and host["memory"] is not None
            else None
        ),
        memswap_limit=(
            f"{host['memory']}mb"
            if "memory" in host and host["memory"] is not None
            else None
        ),
        dimage=DIND_IMAGE,
        privileged=True,
        dcmd="dockerd",
        volumes=[getConfig()["repoAbsolutePath"] + "/docker/files:/home/docker/files"]
    )
