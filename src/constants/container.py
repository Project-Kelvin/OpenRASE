"""
Defines constants related to Docker.
"""

from shared.utils.container import getRegistryContainerTag


TAG = getRegistryContainerTag()
SERVER_IMAGE = f"{TAG}/server:latest"
DIND = f"{TAG}/dind:latest"
