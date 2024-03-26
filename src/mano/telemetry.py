"""
Defines the Telemetry class.
"""


from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from fcntl import ioctl
from json import dumps
from os import listdir
from re import match
import socket
from struct import pack, unpack
from sys import maxsize
from typing import Any
from urllib.request import HTTPHandler, Request, build_opener
import requests
from shared.models.embedding_graph import VNF, EmbeddingGraph, VNFEntity
from shared.models.topology import Host, Topology
from shared.utils.config import getConfig
from shared.utils.container import doesContainerExist
from mininet.net import Mininet
from mininet.util import quietRun
from constants.container import MININET_PREFIX
from constants.notification import EMBEDDING_GRAPH_DEPLOYED
from mano.notification_system import NotificationSystem, Subscriber
from models.telemetry import HostData, SwitchData
from utils.container import connectToDind
from utils.embedding_graph import traverseVNF
from docker import DockerClient, from_env
from docker.models.containers import Container
import time


SFLOW_CONTAINER: str = "sflow"
SFLOW_IMAGE: str = "sflow/sflow-rt"


class Telemetry(Subscriber):
    """
    Class that collects and sends telemetry data of the topology.
    """

    _topology: Topology = None
    _vnfsInHosts: "dict[str, list[VNFEntity]]" = {}
    _sfcHostByIPs: "dict[str, tuple[str, str]]" = {}

    def __init__(self, topology: Topology, sfcHostByIPs: "dict[str, tuple[str, str]]") -> None:
        """
        Constructor for the class.

        Parameters:
            topology (Topology): The topology of the network.
            sfcHostByIPs (dict[str, tuple[str, str]]): The SFC hosts by IPs.
        """

        self._sfcHostByIPs = sfcHostByIPs
        self._topology = topology
        NotificationSystem.subscribe(EMBEDDING_GRAPH_DEPLOYED, self)
        self._startSflow()

    def _startSflow(self):
        """
        Start sflow monitoring.
        """

        client: DockerClient = from_env()

        if doesContainerExist(SFLOW_CONTAINER):
            client.containers.get(SFLOW_CONTAINER).remove(force=True)

        client.containers.run(
            SFLOW_IMAGE,
            detach=True,
            name=SFLOW_CONTAINER,
            ports={
                8008: 8008,
                6343: "6343/udp"
            }
        )

    @classmethod
    def runSflow(cls):
        """
        Run sflow.py script sourced from the sflow-rt package.
        This allows sflow to monitor the Mininet topology.
        """

        cls._startSflow(cls)

        def wrapper(fn):
            def getIfInfo(dst):
                is64bits: bool = maxsize > 2**32
                structSize: int = 40 if is64bits else 32
                s: socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                maxPossible: int = 8  # initial value
                while True:
                    # pylint: disable=redefined-builtin
                    bytes: int = maxPossible * structSize
                    names: "array[int]" = array('B')
                    for i in range(0, bytes):
                        names.append(0)
                    outBytes: Any = unpack('iL', ioctl(
                        s.fileno(),
                        0x8912,  # SIOCGIFCONF
                        pack('iL', bytes, names.buffer_info()[0])
                    ))[0]
                    if outBytes == bytes:
                        maxPossible *= 2
                    else:
                        break
                s.connect((dst, 0))
                ip: Any = s.getsockname()[0]
                for i in range(0, outBytes, structSize):
                    addr: str = socket.inet_ntoa(names[i+20:i+24])
                    if addr == ip:
                        name: "array[int]" = names[i:i+16]
                        try:
                            name = name.tobytes().decode('utf-8')
                        except AttributeError:
                            name = name.tostring()
                        name = name.split('\0', 1)[0]

                        return (name, addr)

            def configSFlow(net, collector, ifname, sampling, polling):
                sflow: str = ("ovs-vsctl -- --id=@sflow create sflow "
                              f"agent={ifname} target={collector} sampling={sampling} polling={polling} --")
                for s in net.switches:
                    sflow += f" -- set bridge {s} sflow=@sflow"
                quietRun(sflow)

            def sendTopology(net, agent, collector):
                topo = {'nodes': {}, 'links': {}}
                for s in net.switches:
                    topo['nodes'][s.name] = {'agent': agent, 'ports': {}}
                path = '/sys/devices/virtual/net/'
                for child in listdir(path):
                    parts = match('(^.+)-(.+)', child)
                    if parts is None:
                        continue
                    if parts.group(1) in topo['nodes']:
                        ifindex = open(path+child+'/ifindex',
                                       encoding="utf").read().split('\n', 1)[0]
                        topo['nodes'][parts.group(1)]['ports'][child] = {
                            'ifindex': ifindex}
                i = 0
                for s1 in net.switches:
                    j = 0
                    for s2 in net.switches:
                        if j > i:
                            intfs: Any = s1.connectionsTo(s2)
                            for intf in intfs:
                                linkName = f"{s1.name}-{s2.name}"
                                topo['links'][linkName] = {
                                    'node1': s1.name,
                                    'port1': intf[0].name,
                                    'node2': s2.name,
                                    'port2': intf[1].name
                                }
                        j += 1
                    i += 1

                request = Request(
                    f'http://{collector}:8008/topology/json', data=dumps(topo).encode('utf-8'))

                request.add_header('Content-Type', 'application/json')
                request.get_method = lambda: 'PUT'
                opener = build_opener(HTTPHandler)
                opener.open(request)

            def result(*args, **kwargs):
                res = fn(*args, **kwargs)
                net = args[0]
                ip = from_env().containers.get(
                    SFLOW_CONTAINER).attrs["NetworkSettings"]["IPAddress"]
                sampling = 10
                polling = 10
                (ifname, agent) = getIfInfo(ip)
                configSFlow(net, ip, ifname, sampling, polling)
                sendTopology(net, agent, ip)

                return res

            return result

        setattr(Mininet, 'start', wrapper(Mininet.__dict__['start']))

    def getHostData(self) -> HostData:
        """
        Get the data of the hosts.
        """

        hosts: "list[Host]" = self._topology["hosts"]
        hostDataFutures: HostData = {}
        hostData: HostData = {}

        with ThreadPoolExecutor() as executor:
            for host in hosts:
                client: DockerClient = from_env()
                hostContainer: Container = client.containers.get(
                    f"{MININET_PREFIX}.{host['id']}Node")
                memoryLimit: float = hostContainer.stats(
                    stream=False)["memory_stats"]["limit"]

                hostCPUUsage: Future = executor.submit(self._calculateCPUUsage,
                    host["id"], hostContainer)
                hostMemoryUsage: Future = executor.submit(self._calculateMemoryUsage,
                    hostContainer, memoryLimit)
                hostNetworkUsage: Future = executor.submit(self._calculateNetworkUsage,
                    hostContainer)

                hostDataFutures[host["id"]] = {
                    "cpuUsage": hostCPUUsage,
                    "memoryUsage": hostMemoryUsage,
                    "networkUsage": hostNetworkUsage,
                    "vnfs": {}
                }

                dindClient: DockerClient = connectToDind(f"{host['id']}Node")

                if host["id"] in self._vnfsInHosts and len(self._vnfsInHosts[host["id"]]) > 0:
                    for vnf in self._vnfsInHosts[host["id"]]:
                        container: Container = dindClient.containers.get(
                            vnf["name"])
                        vnfCPUUsage: Future = executor.submit(self._calculateCPUUsage,
                            host["id"], container)
                        vnfMemoryUsage: Future = executor.submit(self._calculateMemoryUsage,
                            container, memoryLimit)
                        vnfNetworkUsage: Future = executor.submit(self._calculateNetworkUsage,
                            container)
                        hostDataFutures[host["id"]]["vnfs"][vnf["name"]] = {
                            "cpuUsage": vnfCPUUsage,
                            "memoryUsage": vnfMemoryUsage,
                            "networkUsage": vnfNetworkUsage
                        }

            for hostKey, host in hostDataFutures.items():
                hostData[hostKey] = {
                    "cpuUsage": hostDataFutures[hostKey]["cpuUsage"].result(),
                    "memoryUsage": hostDataFutures[hostKey]["memoryUsage"].result(),
                    "networkUsage": hostDataFutures[hostKey]["networkUsage"].result(),
                    "vnfs": {}
                }

                if hostDataFutures[hostKey]["vnfs"] != {}:
                    for vnfKey, vnf in hostDataFutures[hostKey]["vnfs"].items():
                        hostData[hostKey]["vnfs"][vnfKey] = {
                            "cpuUsage": hostDataFutures[hostKey]["vnfs"][vnfKey]["cpuUsage"].result(),
                            "memoryUsage": hostDataFutures[hostKey]["vnfs"][vnfKey]["memoryUsage"].result(),
                            "networkUsage": hostDataFutures[hostKey]["vnfs"][vnfKey]["networkUsage"].result()
                        }


        timestamp: int = int(time.time())
        hostData["timestamp"] = timestamp

        return hostData

    def _calculateCPUUsage(self, host: str, container: Container) -> "tuple[float, float, float]":
        """
        Calculate the CPU usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            "tuple[float, float, float]": The CPU usage of the container (used CPU, remaining CPU, usage percentage).
        """

        stats: Any = container.stats(stream=False)
        # no. of CPUs in the machine.
        totalCPUs: int = stats["cpu_stats"]["online_cpus"]

        # host CPU
        cpu: float = 0.0

        for hostNode in self._topology["hosts"]:
            if hostNode["id"] == host:
                cpu = hostNode["cpu"] if "cpu" in hostNode else totalCPUs

        ratio: float = totalCPUs/cpu

        # CPU usage ratio.
        cpuDelta: float = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
            stats["precpu_stats"]["cpu_usage"]["total_usage"]
        systemDelta: float = stats["cpu_stats"]["system_cpu_usage"] - \
            stats["precpu_stats"]["system_cpu_usage"]
        if systemDelta > 0.0:
            # Removed multiplier to prevent usage exceeding 100%.
            # See: https://github.com/docker/cli/issues/2134
            usagePercentage: float = cpuDelta / systemDelta * ratio * 100.0
            usedCPU: float = cpuDelta/systemDelta * totalCPUs
            remainingCPU: float = totalCPUs - usedCPU

            return (usedCPU, remainingCPU, usagePercentage)
        else:
            return (0.0, 0.0, 0.0)


    def _calculateMemoryUsage(self, container: Container, memoryLimit: float) -> "tuple[float, float, float]":
        """
        Calculate the memory usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            tuple[float, float, float]:
            The memory usage of the container (used memory, remaining memory, usage percentage).
        """

        stats: Any = container.stats(stream=False)
        memUsage: float = stats["memory_stats"]["usage"]

        return (memUsage, memoryLimit - memUsage, memUsage / memoryLimit * 100.0)

    def _calculateNetworkUsage(self, container: Container) -> "tuple[float, float]":
        """
        Calculate the network usage of the container.

        Parameters:
            container (Any): The container.

        Returns:
            tuple[float, float]: The network usage of the container (Rx, Tx).
        """
        stats: Any = container.stats(stream=False)

        rx, tx = 0, 0
        for network in stats["networks"]:
            rx += stats["networks"][network]["rx_bytes"]
            tx += stats["networks"][network]["tx_bytes"]

        return rx/1024, tx/1024

    def getSwitchData(self) -> SwitchData:
        """
        Get the data of the switches.
        """

        sflow: Container = from_env().containers.get(SFLOW_CONTAINER)
        ip: str = sflow.attrs["NetworkSettings"]["IPAddress"]
        sflowGatewayIP = sflow.attrs["NetworkSettings"]["Gateway"]
        sflowUrl = f"http://{ip}:8008"

        timeout: int = getConfig()["general"]["requestTimeout"]

        topo: Any = requests.get(
            f"{sflowUrl}/topology/json", timeout=timeout).json()

        ports = {}

        for node in topo["nodes"]:
            ports.update(topo["nodes"][node]["ports"])

        ifindex = {}

        for key, value in ports.items():
            ifindex[value["ifindex"]] = key

        portLinks = {}
        for key, value in topo["links"].items():
            portLinks[value["port1"]] = key
            portLinks[value["port2"]] = key

        requestData = {
            "value": "bytes",
            "keys": "ipsource,ipdestination"
        }

        requestHeaders = {
            "Content-Type": "application/json",
            "accept": "*/*"
        }

        # pylint: disable=invalid-name
        SRC_DST: str = "srcdst"
        resp: Any = requests.get(
            f"{sflowUrl}/flow/json", timeout=timeout)
        if SRC_DST not in resp.json():
            # srcdst flow not enabled. Enabling it.
            requests.put(f"{sflowUrl}/flow/{SRC_DST}/json",
                         json=requestData, headers=requestHeaders, timeout=timeout)

        switchData: SwitchData = {
            "ipSrcDst": [],
            "inflow": [],
            "outflow": []
        }

        def getPortLink(src: str) -> str:
            """
            Get the port link.

            Parameters:
                src (str): The source.

            Returns:
                str: The port link.
            """

            if ifindex[src] in portLinks:
                return portLinks[ifindex[src]]
            else:
                return ifindex[src]

        resp: Any = requests.get(
            f"{sflowUrl}/dump/{sflowGatewayIP}/{SRC_DST}/json", timeout=timeout)

        for metric in resp.json():
            for key in metric["topKeys"]:
                ipSrcDstStr: "list[str]" = key["key"].split(",")
                srcDst: "tuple[str, str]" = (
                    self._sfcHostByIPs[ipSrcDstStr[0]], self._sfcHostByIPs[ipSrcDstStr[1]])

                switchData["ipSrcDst"].append({
                    "ipSrcDst": srcDst,
                    "interface": getPortLink(metric["dataSource"]),
                    "value": key["value"]
                })

        resp: Any = requests.get(
            f"{sflowUrl}/dump/{sflowGatewayIP}/ifinoctets/json", timeout=timeout)

        for metric in resp.json():
            if metric["dataSource"] in ifindex:
                switchData["inflow"].append({
                    "interface": getPortLink(metric["dataSource"]),
                    "value": metric["metricValue"]
                })

        resp: Any = requests.get(
            f"{sflowUrl}/dump/{sflowGatewayIP}/ifoutoctets/json", timeout=timeout)

        for metric in resp.json():
            if metric["dataSource"] in ifindex:
                switchData["outflow"].append({
                    "interface": getPortLink(metric["dataSource"]),
                    "value": metric["metricValue"]
                })

        timeStamp: int = int(time.time())
        switchData["timestamp"] = timeStamp

        return switchData

    def _mapVNFsToHosts(self, embeddingGraph: EmbeddingGraph) -> None:
        """
        Map the VNFs to the hosts.

        Parameters:
            embeddingGraph (EmbeddingGraph): The embedding graph.
        """

        vnfs: VNF = embeddingGraph["vnfs"]

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

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        """
        Receive a notification.

        Parameters:
            topic (str): The topic of the notification.
            args (list[Any]): The arguments of the notification.
        """

        if topic == EMBEDDING_GRAPH_DEPLOYED:
            self._mapVNFsToHosts(args[0])
