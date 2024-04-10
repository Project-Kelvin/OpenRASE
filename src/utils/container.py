"""
Defines Docker related utils.
"""

from docker import from_env, DockerClient
from docker.models.containers import Container
from constants.container import DIND_TCP_PORT, MININET_PREFIX


def getContainerIP(hostName: str, isMnContainer: bool = True) -> str:
    """
    Get the IP address of a container.

    Parameters:
        hostName (str): The host name of the container.
        isMnContainer (bool): Whether the container is a Mininet container.

    Returns:
        str: The IP address of the container.
    """

    client: DockerClient = from_env()
    hostContainer: Container = client.containers.get(
        f"{MININET_PREFIX}.{hostName}" if isMnContainer else hostName)
    hostIP: str = hostContainer.attrs["NetworkSettings"]["IPAddress"]

    return hostIP


def connectToDind(hostName: str, isMnContainer: bool = True) -> DockerClient:
    """
    Connect to DIND container and return the Docker client.

    Parameters:
        hostName (str): The name of the host.
        isMnContainer (bool): Whether the container is a Mininet container.

    Returns:
        DockerClient: The Docker client.
    """

    hostIP: str = getContainerIP(hostName, isMnContainer)

    dindClient: DockerClient = DockerClient(
        base_url=f"tcp://{hostIP}:{DIND_TCP_PORT}")

    return dindClient

def waitTillContainerReady(name: str, isMnContainer: bool = True) -> None:
    """
    Wait until the container is ready.

    Parameters:
        name (str): The name of the container.
        isMnContainer (bool): Whether the container is a Mininet container.
    """

    isReady: bool = False

    while not isReady:
        try:
            client: DockerClient = connectToDind(name, isMnContainer)
            client.containers.list()
            isReady= True
        # pylint: disable=broad-except
        except Exception:
            pass
