"""
Defines constants related to Docker.
"""

from utils.docker import getRegistryContainerTag


TAG = getRegistryContainerTag()
SERVER_IMAGE = f"{TAG}/server:latest"
DIND = f"{TAG}/dind:latest"
