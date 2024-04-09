"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from threading import Thread, Lock
import requests
from shared.models.embedding_graph import VNF, EmbeddingGraph, VNFEntity
from shared.models.topology import Host as TopoHost
from shared.utils.config import getConfig
from shared.utils.container import getVNFContainerTag
from constants.notification import EMBEDDING_GRAPH_DELETED, EMBEDDING_GRAPH_DEPLOYED
from constants.topology import SERVER, SFCC
from mano.infra_manager import InfraManager
from mano.notification_system import NotificationSystem
from docker import DockerClient
from utils.container import connectToDind, getContainerIP
from utils.embedding_graph import traverseVNF


class VNFManager():
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    _infraManager: InfraManager = None
    _embeddingGraphs: "dict[str, EmbeddingGraph]" = {}
    _ports: "dict[str, int]" = {}
    _portLock = None
    _infraLock = None

    def __init__(self, infraManager: InfraManager) -> None:
        """
        Constructor for the class.

        Parameters:
            infraManager (InfraManager): The infrastructure manager.
        """

        self._infraManager = infraManager
        self._portLock: Lock = Lock()
        self._infraLock: Lock = Lock()

    def _deployEmbeddingGraph(self, eg: EmbeddingGraph) -> None:
        """
        Deploy the embedding graph.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to be deployed.
        """

        if eg["sfcID"] in self._embeddingGraphs:
            self._deleteEmbeddingGraph(self._embeddingGraphs[eg["sfcID"]])

        with self._infraLock:
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
                if vnf["id"] in sharedVolumes:
                    for vol in sharedVolumes[vnf["id"]]:
                        volumes[vol.split(":")[0]] = {
                            "bind": vol.split(":")[1],
                            "mode": "rw"
                        }

                port: int = 5000
                with self._portLock:
                    if host["id"] in self._ports:
                        self._ports[host["id"]] += 1
                    else:
                        self._ports[host["id"]] = 5000
                    port = self._ports[host["id"]]

                dindClient.containers.run(
                    getVNFContainerTag(vnf["id"]),
                    detach=True,
                    name=vnfName,
                    volumes=volumes,
                    ports={"80/tcp": port},
                )

                vnf["ip"] = f"{getConfig()['sff']['network1']['hostIP']}:{port}"

        def traverseCallback(vnfs: VNF, _depth: int) -> None:
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

        print(updatedEG)
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
                dindClient: DockerClient = connectToDind(f"{host['id']}Node")

                dindClient.containers.get(vnfName).remove(force=True)

        def traverseCallback(vnfs: VNF, _depth: int) -> None:
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
