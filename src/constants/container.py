"""
Defines constants related to Docker.
"""

from shared.utils.container import getRegistryContainerTag


TAG = getRegistryContainerTag()
SERVER_IMAGE = f"{TAG}/server:latest"
DIND_IMAGE = f"{TAG}/dind:latest"
DIND_TCP_PORT = 2375
DIND_NETWORK1 = "dind_network1"
DIND_NETWORK2 = "dind_network2"
DIND_NW1_IP = "192.168.0.0/24"
DIND_NW2_IP = "172.16.0.0/24"
SFCC_IMAGE = f"{TAG}/sfcc:latest"
MININET_PREFIX = "mn"
SFF_IMAGE = f"{TAG}/sff:latest"
SFF = "sff"
SFF_IP1 = "192.168.0.2"
SFF_IP2 = "172.16.0.2"
