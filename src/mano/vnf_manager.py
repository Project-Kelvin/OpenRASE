"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from time import sleep
from typing import Any, Tuple, TypedDict
from threading import Thread
import requests
from shared.models.forwarding_graph import VNF, ForwardingGraph, VNFEntity
from shared.models.topology import Host as TopoHost
from shared.utils.config import getConfig
from shared.utils.container import getVNFContainerTag
from constants.notification import FORWARDING_GRAPH_DEPLOYED, SFF_DEPLOYED, TOPOLOGY_INSTALLED
from constants.topology import SERVER, SFCC
from constants.container import DIND_NETWORK1, \
    DIND_NETWORK2, DIND_NW1_IP, DIND_NW2_IP, \
    SFF, SFF_IMAGE, SFF_IP1, SFF_IP2
from mano.infra_manager import InfraManager
from mano.notification_system import NotificationSystem, Subscriber
from docker.types import IPAMConfig, IPAMPool
from docker import DockerClient
from utils.container import connectToDind, getContainerIP
from utils.forwarding_graph import traverseVNF


class VNFManager(Subscriber):
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    _infraManager: InfraManager = None
    _forwardingGraphs: "list[ForwardingGraph]" = []

    def __init__(self, infraManager: InfraManager) -> None:
        """
        Constructor for the class.

        Parameters:
            infraManager (InfraManager): The infrastructure manager.
        """

        self._infraManager = infraManager
        NotificationSystem.subscribe(TOPOLOGY_INSTALLED, self)

    def _deploySFF(self):
        """
        Deploy the SFF.
        """

        hostIPs: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = self._infraManager.getHostIPs()
        threads: "list[Thread]" = []

        def deploySFFinNode(host: str):
            dindClient: DockerClient = connectToDind(host)
            dindClient.networks.create(DIND_NETWORK1,
                                       ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=DIND_NW1_IP)]))
            dindClient.networks.create(DIND_NETWORK2,
                                       ipam=IPAMConfig(pool_configs=[IPAMPool(subnet=DIND_NW2_IP)]))

            container: Any = dindClient.containers.run(
                SFF_IMAGE,
                detach=True,
                name=SFF,
                ports={80: 80}
            )

            dindClient.networks.get(DIND_NETWORK1).connect(
                container.id, ipv4_address=SFF_IP1)
            dindClient.networks.get(DIND_NETWORK2).connect(
                container.id, ipv4_address=SFF_IP2)

        for host in hostIPs:
            if host not in (SERVER, SFCC):
                thread: Thread = Thread(target=deploySFFinNode, args=(host,))
                thread.start()

                threads.append(thread)

        for thread in threads:
            thread.join()

        sleep(10)
        NotificationSystem.publish(SFF_DEPLOYED)

    def _deployForwardingGraph(self, fg: ForwardingGraph) -> None:
        """
        Deploy the forwarding graph.

        Parameters:
            fg (ForwardingGraph): The forwarding graph to be deployed.
        """

        updatedFG: ForwardingGraph = self._infraManager.assignIPs(fg)
        vnfs: VNF = updatedFG["vnfs"]
        sfcId: str = updatedFG["sfcID"]
        vnfList: "list[str]" = []
        sharedVolumes: "TypedDict[str, list[str]]" = getConfig()[
            "vnfs"]["sharedVolumes"]
        threads: "list[Thread]" = []

        def deployVNF(vnfs: VNF):
            host: TopoHost = vnfs["host"]
            vnf: VNFEntity = vnfs["vnf"]

            vnfList.append(vnf["id"])
            vnfName: str = f"{sfcId}-{vnf['id']}-{len(vnfList)}"
            vnf["name"] = vnfName

            if host["id"] != SERVER:
                dindClient: DockerClient = connectToDind(host["id"])

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

                vnf["ip"] = dindClient.containers.get(
                    container.id).attrs["NetworkSettings"]["Networks"][DIND_NETWORK1]["IPAddress"]

        def traverseCallback(vnfs: VNF) -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnfs (VNF): The VNF.
            """

            thread: Thread = Thread(target=deployVNF, args=(vnfs,))
            thread.start()

            threads.append(thread)

        traverseVNF(vnfs, traverseCallback, shouldParseTerminal=False)

        for thread in threads:
            thread.join()

        self._forwardingGraphs.append(updatedFG)

        sfccIP: str = getContainerIP(SFCC)

        requests.post(
            f"http://{sfccIP}/add-fg",
            json=updatedFG,
            timeout=getConfig()["general"]["requestTimeout"]
        )

        NotificationSystem.publish(FORWARDING_GRAPH_DEPLOYED, updatedFG)

    def deployForwardingGraphs(self, fgs: "list[ForwardingGraph]") -> None:
        """
        Deploy the forwarding graphs.

        Parameters:
            fgs (list[ForwardingGraph]): The forwarding graphs to be deployed.
        """

        threads: "list[Thread]" = []
        for fg in fgs:
            thread: Thread = Thread(
                target=self._deployForwardingGraph, args=(fg,))
            thread.start()

            threads.append(thread)

        for thread in threads:
            thread.join()

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        """
        Receive a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """
        if topic == TOPOLOGY_INSTALLED:
            self._deploySFF()
