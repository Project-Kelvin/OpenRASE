"""
Defines utils associated with Docker.
"""

from docker import DockerClient, from_env
from shared.constants.sfc import SFC_REGISTRY

client: DockerClient = from_env()

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
