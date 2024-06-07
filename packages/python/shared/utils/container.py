"""
Provides util functions related to Docker.
"""

from typing import Any
from shared.constants.sfc import SFC_REGISTRY
from docker import from_env, DockerClient, errors

client: DockerClient = from_env()

def isContainerRunning(name: str)-> bool:
    """
    Check whether the given container is running.

    Parameters:
        name (str): The name of the container to be checked.

    Returns:
        bool: True if the container is running, False otherwise.
    """

    try:
        registry: Any = client.containers.get(name)

        return registry.status == "running"
    except errors.NotFound:
        return False

def doesContainerExist(name: str) -> bool:
    """
    Check whether the given container exists.

    Parameters:
        name (str): The name of the container to be checked.

    Returns:
        bool: True if the container exists, False otherwise.
    """

    try:
        client.containers.get(name)
    except errors.NotFound:
        return False

    return True


def getRegistryContainerIP() -> str:
    """
    Get the IP address of the registry container.

    Returns:
        str: The IP address of the registry container.
    """

    return client.containers.get(SFC_REGISTRY).attrs["NetworkSettings"]["IPAddress"]


def getRegistryContainerTag() -> str:
    """
    Get the tag of the registry container.

    Returns:
        str: The tag of the registry container.
    """

    return f"{getRegistryContainerIP()}:5000"

def getVNFContainerTag(vnf: str) -> str:
    """
    Get the tag of the VNF container.

    Parameters:
        vnf (str): The name of the VNF.

    Returns:
        str: The tag of the VNF container.
    """

    return f"{getRegistryContainerTag()}/{vnf}:latest"
