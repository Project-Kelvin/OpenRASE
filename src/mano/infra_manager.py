"""
Defines the class that corresponds to teh Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from typing import Any, Tuple, TypedDict
from shared.models.config import Config
from shared.utils.config import getConfig

from shared.utils.ip import generateLocalNetworkIP
from mininet.node import Ryu, Host
from mininet.net import Containernet
from shared.models.forwarding_graph import VNF, ForwardingGraph
from shared.models.topology import Topology


class InfraManager():
    """
    Class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
    """

    net: Any = None
    ryu: Ryu = None
    topology: Topology = None
    networkIPs: "list[IPv4Network]" = []
    hostGateways: "TypedDict[str, IPv4Address]" = {}
    hosts: "list[Host]" = []
    switches: "list[Host]" = []

    def __init__(self) -> None:
        """
        Constructor for the class.
        """
        self.net = Containernet()
        self.ryu = Ryu('ryu', ryuArgs="ryu.app.rest_router",
                       command="ryu-manager")
        self.net.addController(self.ryu)
        self.net.start()

    def generateIP(self) -> "Tuple[IPv4Address, IPv4Address]":
        """
        Generates an IP address for the network.

        Returns:
        str: The generated IP address and the mask.
        """

        config: Config = getConfig()
        mask: int = config['ipRange']['mask']

        ip: IPv4Network = generateLocalNetworkIP(mask, self.networkIPs)
        self.networkIPs.append(ip)
        ip1: IPv4Address = next(ip.hosts())
        ip2: IPv4Address = next(ip.hosts())

        return (ip1, ip2)

    def installTopology(self, topology: Topology) -> None:
        """
        Spins up the provided topology virtually using Mininet (Containernet).

        Parameters:
        topology (Topology): The topology to be spun up.
        """

        self.topology = topology
        cpuPeriod: int = 100000  # Docker default

        for host in topology['hosts']:
            ip: "Tuple[IPv4Address, IPv4Address]" = self.generateIP()
            self.hostGateways[host['id']] = ip[0]
            self.hosts.append(self.net.addHost(
                host['id'],
                ip=ip[1],
                cpu_quota=host['cpu'] * cpuPeriod,
                mem_limit=host['memory'],
                memswap_limit=host['memory']
            ))

        for switch in topology['switches']:
            self.switches.append(self.net.addSwitch(
                switch['id'], protocols='OpenFlow13'))

        for link in topology['links']:
            self.net.addLink(
                link['source'], link['destination'], bw=link['bandwidth'])

    def getNode(self, name: str) -> Host:
        """
        Returns the node with the given name in the topology.

        Parameters:
        name (str): The name of the node to be returned.

        Returns:
        Host: The node with the given name in the topology.
        """

        return self.net.get(name)

    def getSwitches(self) -> Any:
        """
        Returns the switches in the topology.
        """

        return self.switches

    def getHosts(self) -> Any:
        """
        Returns the hosts in the topology.
        """

        return self.hosts

    def assignIPs(self, fg: ForwardingGraph) -> ForwardingGraph:
        """
        Assigns IPs to the hosts in the topology.

        Parameters:
        fg (ForwardingGraph): The forwarding graph to be used to assign IPs.

        Returns:
        ForwardingGraph: The forwarding graph with the IPs assigned.
        """

        vnfs: VNF = fg['vnfs']

        def traverseVNF(vnfs: VNF):
            """
            Traverses the VNFs in the forwarding graph and assigns IPs to the hosts.
            """

            shouldContinue: bool = True

            while shouldContinue:
                ipAddr: str = self.generateIP()
                vnfs['host']['ip'] = ipAddr
                self.net.get(vnfs['host']['id']).cmd(
                    f"ip addr add {ipAddr} dev {vnfs['host']['id']}-eth0")

                if isinstance(vnfs['next'], list):
                    for nextVnf in vnfs['next']:
                        traverseVNF(nextVnf)

                    shouldContinue = False
                else:
                    vnfs = vnfs['next']

                if vnfs == "terminal":
                    shouldContinue = False

        traverseVNF(vnfs)

        return fg
