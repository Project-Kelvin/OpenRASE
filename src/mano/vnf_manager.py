"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from typing import Any, Tuple, TypedDict
from threading import Thread
from shared.models.forwarding_graph import VNF, ForwardingGraph, VNFEntity
from shared.models.topology import Host as TopoHost
from shared.utils.config import getConfig
from shared.utils.container import getVNFContainerTag
from constants.topology import SERVER, SFCC, TERMINAL
from constants.container import DIND_NETWORK1, \
    DIND_NETWORK2, DIND_NW1_IP, DIND_NW2_IP, DIND_TCP_PORT, MININET_PREFIX, \
    SFF, SFF_IMAGE, SFF_IP1, SFF_IP2
from mano.infra_manager import InfraManager
from docker.types import IPAMConfig, IPAMPool
from docker.models.containers import Container
from docker import DockerClient, from_env

class VNFManager():
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    infraManager: InfraManager = None
    forwardingGraphs: "list[ForwardingGraph]" = []

    def __init__(self, infraManager: InfraManager) -> None:
        """
        Constructor for the class.

        Parameters:
            infraManager (InfraManager): The infrastructure manager.
        """

        self.infraManager = infraManager

    def connectToDind(self, hostName: str) -> DockerClient:
        """
        Get the IP address of the host.

        Parameters:
            hostName (str): The name of the host.

        Returns:
            DockerClient: The Docker client.
        """
        client: DockerClient = from_env()
        hostContainer: Container = client.containers.get(
            f"{MININET_PREFIX}.{hostName}")
        hostIP: str = hostContainer.attrs["NetworkSettings"]["IPAddress"]

        dindClient: DockerClient = DockerClient(
            base_url=f"tcp://{hostIP}:{DIND_TCP_PORT}")

        return dindClient

    def deploySFF(self):
        """
        Deploy the SFF.
        """

        hostIPs: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = self.infraManager.getHostIPs()
        threads: "list[Thread]" = []

        def deploySFFinNode(host: str):
            dindClient: DockerClient = self.connectToDind(host)
            dindClient.networks.create(DIND_NETWORK1,
                                        ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=DIND_NW1_IP)]))
            dindClient.networks.create(DIND_NETWORK2,
                                        ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=DIND_NW2_IP)]))

            container: Any = dindClient.containers.run(
                SFF_IMAGE,
                detach=True,
                name=SFF,
                ports={80:80}
            )

            dindClient.networks.get(DIND_NETWORK1).connect(container.id, ipv4_address=SFF_IP1)
            dindClient.networks.get(DIND_NETWORK2).connect(container.id, ipv4_address=SFF_IP2)

        for host in hostIPs:
            if host != SERVER and host != SFCC:
                thread: Thread = Thread(target=deploySFFinNode, args=(host,))
                thread.start()

                threads.append(thread)

        for thread in threads:
            thread.join()

    def deployForwardingGraph(self, fg: ForwardingGraph) -> None:
        """
        Deploy the forwarding graph.

        Parameters:
            fg (ForwardingGraph): The forwarding graph to be deployed.
        """

        updatedFG: ForwardingGraph = self.infraManager.assignIPs(fg)
        vnfs: VNF = updatedFG["vnfs"]
        sfcId: str = updatedFG["sfcID"]
        vnfList: "list[str]" = []
        sharedVolumes: "TypedDict[str, list[str]]" = getConfig()["vnfs"]["sharedVolumes"]
        threads: "list[Thread]" = []

        def deployVNF(vnfs: VNF):
            host: TopoHost = vnfs["host"]
            vnf: VNFEntity = vnfs["vnf"]

            vnfList.append(vnf["id"])
            vnfName: str = f"{sfcId}-{vnf['id']}-{len(vnfList)}"
            vnf["name"] = vnfName

            if host["id"] != SERVER:
                dindClient: DockerClient = self.connectToDind(host["id"])

                volumes = {}
                for vol in sharedVolumes[vnf["id"]]:
                    volumes[vol.split(":")[0]] = {
                        "bind": vol.split(":")[1],
                        "mode": "rw"
                    }

                container: Any = dindClient.containers.run(
                    getVNFContainerTag(vnf["id"]),
                    detach=True,
                    name=vnfName,
                    volumes=volumes,
                    network_mode=DIND_NETWORK1
                )

                dindClient.networks.get(DIND_NETWORK2).connect(container.id)

                vnf["ip"] = container.attrs["NetworkSettings"]["Networks"][DIND_NETWORK1]["IPAddress"]

        def traverseVNF(vnfs: VNF):
            """
            Traverse the VNFs in the forwarding graph and deploys them.

            Parameters:
                vnfs (VNF): The VNF to be traversed.
            """

            nonlocal vnfList

            shouldContinue: bool = True

            while shouldContinue:
                if vnfs["next"] == TERMINAL:
                    break

                thread: Thread = Thread(target=deployVNF, args=(vnfs,))
                thread.start()

                threads.append(thread)

                if isinstance(vnfs['next'], list):
                    for nextVnf in vnfs['next']:
                        traverseVNF(nextVnf)

                    shouldContinue = False
                else:
                    vnfs = vnfs['next']

        for thread in threads:
            thread.join()

        traverseVNF(vnfs)
        self.forwardingGraphs.append(updatedFG)
        self.infraManager.startCLI()
        self.infraManager.stopNetwork()
