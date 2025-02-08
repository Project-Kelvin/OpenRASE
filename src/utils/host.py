"""
This defines the logic to add host to the Mininet environment.
"""

from typing import Any
from docker import DockerClient
from shared.models.config import Config
from shared.models.topology import Host
from mininet.node import Host as MininetHost
from shared.utils.config import getConfig

from constants.container import CPU_PERIOD, DIND_IMAGE, SFF_IMAGE, SFF_TX_IMAGE
from utils.container import connectToDind

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

def addSFF(host: Host, net: Any) -> MininetHost:
    """
    Add a SFF node to the Mininet environment.

    Parameters:
        host (Host): The host to add.
        net (Any): The Mininet environment.

    Returns:
        MininetHost: The SFF node.
    """

    return net.addDocker(
        host["id"],
        ip=f"{getConfig()['sff']['network1']['sffIP']}/{getConfig()['sff']['network1']['mask']}",
        dimage=DIND_IMAGE,
        dcmd="dockerd",
        defaultRoute=f"dev {host['id']}-eth1",
        volumes=[getConfig()["repoAbsolutePath"] + "/docker/files:/home/docker/files"],
        privileged=True
    )

def addSFFEnds(host: str) -> None:
    """
    Add the SFF ends to the Mininet environment.

    Parameters:
        host (str): The host to add.
    """

    config: Config = getConfig()
    dindClient: DockerClient = connectToDind(host)
    dindClient.containers.run(
        SFF_IMAGE,
        detach=True,
        name="rx",
        volumes=["/home/docker/files/node-sff-rx/shared/node-logs:/home/OpenRASE/apps/sff-rx/node-logs"],
        ports={f'{config["sff"]["port"]}/tcp': config["sff"]["port"]},
        restart_policy={"Name": "on-failure"},
    )
    dindClient.containers.run(
        SFF_TX_IMAGE,
        detach=True,
        name="tx",
        volumes=[
            "/home/docker/files/node-sff-tx/shared/node-logs:/home/OpenRASE/apps/sff-tx/node-logs"
        ],
        ports={f'{config["sff"]["txPort"]}/tcp': config["sff"]["txPort"]},
        restart_policy={"Name": "on-failure"},
    )
