"""
Provides util functions related to Docker.
"""

from typing import Any
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

        return registry.status != "running"
    except errors.NotFound:
        return False
