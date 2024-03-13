"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from threading import Thread
import requests
from shared.models.embedding_graph import VNF, EmbeddingGraph, VNFEntity
from shared.models.topology import Host as TopoHost
from shared.utils.config import getConfig
from shared.utils.container import getVNFContainerTag
from constants.notification import EMBEDDING_GRAPH_DELETED, EMBEDDING_GRAPH_DEPLOYED
from constants.topology import SERVER, SFCC
from mano.infra_manager import InfraManager
from docker import DockerClient
from mano.notification_system import NotificationSystem
from utils.container import connectToDind, getContainerIP
from utils.embedding_graph import traverseVNF


class VNFManager():
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    _infraManager: InfraManager = None
    _embeddingGraphs: "dict[str, EmbeddingGraph]" = {}
    _ports: "dict[str, int]" = {}

    def __init__(self, infraManager: InfraManager) -> None:
        """
        Constructor for the class.

        Parameters:
            infraManager (InfraManager): The infrastructure manager.
        """

        self._infraManager = infraManager

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
                dindClient: DockerClient = connectToDind(f"{host['id']}Node")

                volumes = {}
                for vol in sharedVolumes[vnf["id"]]:
                    volumes[vol.split(":")[0]] = {
                        "bind": vol.split(":")[1],
                        "mode": "rw"
                    }

                if host["id"] in self._ports:
                    self._ports[host["id"]] += 1
                else:
                    self._ports[host["id"]] = 5000

                dindClient.containers.run(
                    getVNFContainerTag(vnf["id"]),
                    detach=True,
                    name=vnfName,
                    volumes=volumes,
                    ports={"80/tcp": f"{self._ports[host['id']]}"},
                )

                vnf["ip"] = f"{getConfig()['sff']['network1']['hostIP']}:{self._ports[host['id']]}"

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
