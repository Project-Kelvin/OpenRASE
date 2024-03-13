"""
Defines constants related to Docker.
"""

from shared.utils.container import getRegistryContainerTag


TAG: str = getRegistryContainerTag()
SERVER_IMAGE: str = f"{TAG}/server:latest"
DIND_IMAGE: str = f"{TAG}/dind:latest"
DIND_TCP_PORT: str = 2375
DIND_NETWORK1: str = "dind_network1"
DIND_NETWORK2: str = "dind_network2"
SFCC_IMAGE: str = f"{TAG}/sfcc:latest"
MININET_PREFIX: str = "mn"
SFF_IMAGE: str = f"{TAG}/sff:latest"
SFF: str = "sff"
SFF_IP1: str = "192.168.0.2"
SFF_IP2: str = "172.16.0.2"
CPU_PERIOD: int = 100000
