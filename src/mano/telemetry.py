"""
Defines the Telemetry class.
"""


from typing import Any
from shared.models.forwarding_graph import VNF, ForwardingGraph, VNFEntity
from shared.models.topology import Host, Topology
from constants.container import MININET_PREFIX
from constants.notification import FORWARDING_GRAPH_DEPLOYED
from mano.notification_system import NotificationSystem, Subscriber
from models.telemetry import HostData, SingleHostData, VNFData
from utils.container import connectToDind
from utils.forwarding_graph import traverseVNF
from docker import DockerClient, from_env
from docker.models.containers import Container

class Telemetry(Subscriber):
    """
    Class that collects and sends telemetry data of the topology.
    """

    _topology: Topology = None
    _vnfsInHosts: "dict[str, list[VNFEntity]]" = {}

    def __init__(self, topology: Topology) -> None:
        """
        Constructor for the class.

        Parameters:
            topology (Topology): The topology of the network.
        """

        self._topology = topology
        NotificationSystem.subscribe(FORWARDING_GRAPH_DEPLOYED, self)

    def getHostData(self) -> HostData:
        """
        Get the data of the hosts.
        """

        hosts: "list[Host]" = self._topology["hosts"]
        hostData: HostData = {}

        for host in hosts:
            client: DockerClient = from_env()
            hostContainer: Container = client.containers.get(f"{MININET_PREFIX}.{host['id']}")
            hostCPUUsage: float = self._calculateCPUUsage(host["id"], hostContainer)
            memoryLimit: float = hostContainer.stats(stream=False)["memory_stats"]["limit"]
            hostMemoryUsage: float = self._calculateMemoryUsage(hostContainer, memoryLimit)
            hostNetworkUsage: float = self._calculateNetworkUsage(hostContainer)
            dindClient: DockerClient = connectToDind(host["id"])

            hostData[host["id"]] = {
                "cpuUsage": hostCPUUsage,
                "memoryUsage": hostMemoryUsage,
                "networkUsage": hostNetworkUsage,
                "vnfs": {}
            }

            if len(self._vnfsInHosts[host["id"]]) > 0:
                for vnf in self._vnfsInHosts[host["id"]]:
                    container: Container = dindClient.containers.get(vnf["name"])
                    vnfCPUUsage: float = self._calculateCPUUsage(host["id"], container)
                    vnfMemoryUsage: float = self._calculateMemoryUsage(container, memoryLimit)
                    vnfNetworkUsage: float = self._calculateNetworkUsage(container)
                    hostData[host["id"]]["vnfs"][vnf["name"]] = {
                        "cpuUsage": vnfCPUUsage,
                        "memoryUsage": vnfMemoryUsage,
                        "networkUsage": vnfNetworkUsage
                    }

        return hostData

    def _calculateCPUUsage(self, host: str, container: Container) -> float:
        """
        Calculate the CPU usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            float: The CPU usage of the container.
        """

        stats: Any = container.stats(stream=False)
        # no. of CPUs in the machine.
        noOfCPUs: int = stats["cpu_stats"]["online_cpus"]

        # host CPU
        cpu: float = 0.0

        for hostNode in self._topology["hosts"]:
            if hostNode["id"] == host:
                cpu = hostNode["cpu"]

        ratio: float = noOfCPUs/cpu

        # CPU usage ratio.
        cpuDelta: float = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
            stats["precpu_stats"]["cpu_usage"]["total_usage"]
        systemDelta: float = stats["cpu_stats"]["system_cpu_usage"] - \
            stats["precpu_stats"]["system_cpu_usage"]
        if systemDelta > 0.0:
            # Removed multiplier to prevent usage exceeding 100%.
            # See: https://github.com/docker/cli/issues/2134
            return cpuDelta / systemDelta * ratio * 100.0
        else:
            return 0.0

    def _calculateMemoryUsage(self, container: Container, memoryLimit: float) -> float:
        """
        Calculate the memory usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            float: The memory usage of the container.
        """

        stats: Any = container.stats(stream=False)

        return stats["memory_stats"]["usage"] / memoryLimit * 100.0

    def _calculateNetworkUsage(self, container: Container) -> float:
        """
        Calculate the network usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            float: The network usage of the container.
        """
        stats: Any = container.stats(stream=False)

        rx, tx = 0, 0
        for network in stats["networks"]:
            rx += stats["networks"][network]["rx_bytes"]
            tx += stats["networks"][network]["tx_bytes"]

        return rx/1024, tx/1024

    def getSwitchData(self) -> None:
        """
        Get the data of the switches.
        """
        pass

    def _mapVNFsToHosts(self, forwardingGraph: ForwardingGraph) -> None:
        """
        Map the VNFs to the hosts.

        Parameters:
            forwardingGraph (ForwardingGraph): The forwarding graph.
        """

        vnfs: VNF = forwardingGraph["vnfs"]

        def traverseCallback(vnf: VNF) -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnf (VNF): The VNF being parsed.
            """

            if vnf["host"]["id"] in self._vnfsInHosts:
                self._vnfsInHosts[vnf["host"]["id"]].append(vnf["vnf"])
            else:
                self._vnfsInHosts[vnf["host"]["id"]] = [vnf["vnf"]]

        traverseVNF(vnfs, traverseCallback, shouldParseTerminal=False)

    def receive(self, topic, *args: "list[Any]") -> None:
        """
        Receive a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """

        if topic == FORWARDING_GRAPH_DEPLOYED:
            self._mapVNFsToHosts(args[0])
