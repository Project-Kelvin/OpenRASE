"""
Defines Docker related utils.
"""

from docker import from_env, DockerClient
from docker.models.containers import Container
from constants.container import DIND_TCP_PORT, MININET_PREFIX


def getContainerIP(hostName: str) -> str:
    """
    Get the IP address of a container.

    Parameters:
        hostName (str): The host name of the container.

    Returns:
        str: The IP address of the container.
    """

    client: DockerClient = from_env()
    hostContainer: Container = client.containers.get(
        f"{MININET_PREFIX}.{hostName}")
    hostIP: str = hostContainer.attrs["NetworkSettings"]["IPAddress"]

    return hostIP

def connectToDind(hostName: str) -> DockerClient:
    """
    Connect to DIND container and return teh Docker client.

    Parameters:
        hostName (str): The name of the host.

    Returns:
        DockerClient: The Docker client.
    """

    hostIP: str = getContainerIP(hostName)

    dindClient: DockerClient = DockerClient(
        base_url=f"tcp://{hostIP}:{DIND_TCP_PORT}")

    return dindClient
