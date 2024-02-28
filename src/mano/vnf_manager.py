"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from time import sleep
from typing import Any, Tuple
from threading import Thread
import requests
from shared.models.embedding_graph import VNF, EmbeddingGraph, VNFEntity
from shared.models.topology import Host as TopoHost
from shared.utils.config import getConfig
from shared.utils.container import getVNFContainerTag
from constants.notification import EMBEDDING_GRAPH_DELETED, EMBEDDING_GRAPH_DEPLOYED, SFF_DEPLOYED, TOPOLOGY_INSTALLED
from constants.topology import SERVER, SFCC
from constants.container import DIND_NETWORK1, \
    DIND_NETWORK2, DIND_NW1_IP, DIND_NW2_IP, \
    SFF, SFF_IMAGE, SFF_IP1, SFF_IP2
from mano.infra_manager import InfraManager
from mano.notification_system import NotificationSystem, Subscriber
from docker.types import IPAMConfig, IPAMPool
from docker import DockerClient
from utils.container import connectToDind, getContainerIP
from utils.embedding_graph import traverseVNF


class VNFManager(Subscriber):
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    _infraManager: InfraManager = None
    _embeddingGraphs: "dict[str, EmbeddingGraph]" = {}
    _vnfHosts: "dict[str, ]"

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

        hostIPs: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = self._infraManager.getHostIPs()
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

    def _deployEmbeddingGraph(self, eg: EmbeddingGraph) -> None:
        """
        Deploy the embedding graph.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to be deployed.
        """

        if eg["sfcID"] in self._embeddingGraphs:
            self._deleteEmbeddingGraph(self._embeddingGraphs[eg["sfcID"]])

        updatedEG: EmbeddingGraph = self._infraManager.embedSFC(eg)
        vnfs: VNF = updatedEG["vnfs"]
        sfcId: str = updatedEG["sfcID"]
        vnfList: "list[str]" = []
        sharedVolumes: "dict[str, list[str]]" = getConfig()[
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

        self._embeddingGraphs[sfcId] = updatedEG

        sfccIP: str = getContainerIP(SFCC)

        requests.post(
            f"http://{sfccIP}/add-eg",
            json=updatedEG,
            timeout=getConfig()["general"]["requestTimeout"]
        )

        NotificationSystem.publish(EMBEDDING_GRAPH_DEPLOYED, updatedEG)

    def _deleteEmbeddingGraph(self, eg: EmbeddingGraph) -> None:
        """
        Delete the embedding graph.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to be deleted.
        """

        NotificationSystem.publish(EMBEDDING_GRAPH_DELETED, eg)
        self._infraManager.deleteSFC(eg)
        vnfs: VNF = eg["vnfs"]
        vnfList: "list[str]" = []
        threads: "list[Thread]" = []

        def deleteVNF(vnfs: VNF):
            host: TopoHost = vnfs["host"]
            vnf: VNFEntity = vnfs["vnf"]

            vnfList.append(vnf["id"])
            vnfName: str = vnf["name"]

            if host["id"] != SERVER:
                dindClient: DockerClient = connectToDind(host["id"])

                dindClient.containers.get(vnfName).remove(force=True)

        def traverseCallback(vnfs: VNF) -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnfs (VNF): The VNF.
            """

            thread: Thread = Thread(target=deleteVNF, args=(vnfs,))
            thread.start()

            threads.append(thread)

        traverseVNF(vnfs, traverseCallback, shouldParseTerminal=False)

        for thread in threads:
            thread.join()


    def deployEmbeddingGraphs(self, egs: "list[EmbeddingGraph]") -> None:
        """
        Deploy the embedding graphs.

        Parameters:
            egs (list[EmbeddingGraph]): The embedding graphs to be deployed.
        """

        threads: "list[Thread]" = []
        for eg in egs:
            thread: Thread = Thread(
                target=self._deployEmbeddingGraph, args=(eg,))
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
