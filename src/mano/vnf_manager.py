"""
Defines the VNFManager class that corresponds to the VNF Manager in the NFV architecture.
"""

from concurrent.futures import Future, ThreadPoolExecutor
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
from docker import DockerClient, from_env
from utils.container import connectToDind, getContainerIP
from utils.embedding_graph import traverseVNF
from utils.tui import TUI


class VNFManager():
    """
    Class that corresponds to the VNF Manager in the NFV architecture.
    """

    def __init__(self, infraManager: InfraManager) -> None:
        """
        Constructor for the class.

        Parameters:
            infraManager (InfraManager): The infrastructure manager.
        """


        self._infraManager: InfraManager = infraManager
        self._portLock: Lock = Lock()
        self._infraLock: Lock = Lock()
        self._embeddingGraphs: "dict[str, EmbeddingGraph]" = {}
        self._ports: "dict[str, int]" = {}
        self._deleteLocks: "dict[str, ]" = {}

    def _deployEmbeddingGraph(self, eg: EmbeddingGraph) -> EmbeddingGraph:
        """
        Deploy the embedding graph.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to be deployed.

        Returns:
            EmbeddingGraph: The updated embedding graph.
        """

        if eg["sfcID"] in self._embeddingGraphs:
            TUI.appendToLog(f"Deleting previous embedding graph {eg['sfcID']}:")
            self._deleteEmbeddingGraph(self._embeddingGraphs[eg["sfcID"]])
            TUI.appendToLog(f"Deleted previous embedding graph {eg['sfcID']}.")

        with self._infraLock:
            TUI.appendToLog(f"  Deploying embedding graph {eg['sfcID']}:")
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

                TUI.appendToLog(f"    Deploying {vnfName} ({getVNFContainerTag(vnf['id'])}) on {host['id']} with IP {getConfig()['sff']['network1']['hostIP']}:{port}.")
                try:
                    dindClient.containers.run(
                        getVNFContainerTag(vnf["id"]),
                        detach=True,
                        name=vnfName,
                        volumes=volumes,
                        ports={"80/tcp": port},
                        restart_policy={"Name": "on-failure"},
                    )
                except Exception as e:
                    TUI.appendToLog(f"    Error deploying {vnfName}: {e}", True)
                    return

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

        TUI.appendToLog(f"    Sending embedding graph {sfcId} to SFCC.")
        requests.post(
            f"http://{sfccIP}/add-eg",
            json=updatedEG,
            timeout=getConfig()["general"]["requestTimeout"]
        )

        TUI.appendToLog(f"  Deployed embedding graph {sfcId} successfully.")

        return updatedEG

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

        TUI.appendToLog("  Deleting VNFs:")
        def deleteVNF(vnfs: VNF):
            host: TopoHost = vnfs["host"]
            vnf: VNFEntity = vnfs["vnf"]

            if host["id"] not in self._deleteLocks:
                self._deleteLocks[host["id"]] = Lock()

            vnfList.append(vnf["id"])
            vnfName: str = vnf["name"]

            if host["id"] != SERVER:
                dindClient: DockerClient = connectToDind(f"{host['id']}Node")

                TUI.appendToLog(f"    Deleting {vnfName}.")
                try:
                    dindClient.containers.get(vnfName).stop()
                    dindClient.containers.get(vnfName).remove(force=True)
                except Exception as e:
                    TUI.appendToLog(f"    Error deleting {vnfName}: {e}", True)

                with self._deleteLocks[host["id"]]:
                    dindClient.volumes.prune()

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

        TUI.appendToLog(f"Deleted embedding graph {eg['sfcID']} successfully.")

    def deleteEmbeddingGraphs(self, egs: "list[EmbeddingGraph]") -> None:
        """
        Delete the embedding graphs.

        Parameters:
            egs (list[EmbeddingGraph]): The embedding graphs to be deleted.
        """

        TUI.appendToLog(f"Deleting {len(egs)} embedding graphs:")

        futures: "list[Future[None]]" = []

        with ThreadPoolExecutor() as executor:
            for eg in egs:
                if(eg["sfcID"] in self._embeddingGraphs):
                    del self._embeddingGraphs[eg["sfcID"]]
                    TUI.appendToLog(f"  {eg['sfcID']}")
                    futures.append(executor.submit(self._deleteEmbeddingGraph, eg))

            for future in futures:
                future.result()

        client: DockerClient = from_env()
        client.containers.prune()
        client.images.prune()
        client.networks.prune()
        client.volumes.prune()


    def deployEmbeddingGraphs(self, egs: "list[EmbeddingGraph]") -> None:
        """
        Deploy the embedding graphs.

        Parameters:
            egs (list[EmbeddingGraph]): The embedding graphs to be deployed.
        """

        TUI.appendToLog(f"Deploying {len(egs)} embedding graphs:")

        futures: "list[Future[EmbeddingGraph]]" = []
        updatedEGs: "list[EmbeddingGraph]" = []

        with ThreadPoolExecutor() as executor:
            for eg in egs:
                TUI.appendToLog(f"  {eg['sfcID']}")
                futures.append(executor.submit(self._deployEmbeddingGraph, eg))

            for future in futures:
                updatedEGs.append(future.result())

        NotificationSystem.publish(EMBEDDING_GRAPH_DEPLOYED, updatedEGs)
