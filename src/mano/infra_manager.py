"""
Defines the class that corresponds to teh Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from time import sleep
from typing import Any, Iterator, Tuple, TypedDict
from shared.models.config import Config
from shared.utils.config import getConfig

from shared.utils.ip import generateLocalNetworkIP
from mininet.node import Ryu, Host
from mininet.net import Containernet
from shared.models.forwarding_graph import VNF, ForwardingGraph
from shared.models.topology import Topology
from mininet.cli import CLI

from sfc.sdn_controller import SDNController

class InfraManager():
    """
    Class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
    """

    net: Any = None
    ryu: Ryu = None
    topology: Topology = None
    networkIPs: "list[IPv4Network]" = []
    hosts: "TypedDict[str, Host]" = {}
    switches: "TypedDict[str, OVSKernelSwitch]" = {}
    sdnController: SDNController = None

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self.sdnController = SDNController(self)
        self.net = Containernet()
        self.ryu = Ryu('ryu', ryuArgs="ryu.app.rest_router",
                       command="ryu-manager")
        self.net.addController(self.ryu)

    def generateIP(self) -> "Tuple[IPv4Network, IPv4Address, IPv4Address]":
        """
        Generate an IP address for the network.

        Returns:
        "Tuple[IPv4Network, IPv4Address, IPv4Address]": The generated IP address and the first two host IPs.
        """

        config: Config = getConfig()
        mask: int = config['ipRange']['mask']

        ip: IPv4Network = generateLocalNetworkIP(mask, self.networkIPs)
        self.networkIPs.append(ip)
        hosts: "Iterator[IPv4Address]" = ip.hosts()
        ip1: IPv4Address = next(hosts)
        ip2: IPv4Address = next(hosts)

        return (ip, ip1, ip2)

    def installTopology(self, topology: Topology) -> None:
        """
        Spin up the provided topology virtually using Mininet (Containernet).

        Parameters:
        topology (Topology): The topology to be spun up.
        """

        self.topology = topology
        cpuPeriod: int = 100000  # Docker default
        config: Config = getConfig()
        hostGateways: "TypedDict[str, IPv4Address]" = {}

        for host in topology['hosts']:
            ip: "Tuple[IPv4Address, IPv4Address]" = self.generateIP()
            hostGateways[host['id']] = ip[1]

            hostNode: Host = self.net.addHost(
                host['id'],
                ip=f"{str(ip[2])}/{config['ipRange']['mask']}",
                cpu_quota=host['cpu'] * cpuPeriod,
                mem_limit=host['memory'],
                memswap_limit=host['memory']
            )
            self.hosts[host["id"]]=hostNode
            hostNode.cmd(f"ip route add default via {str(ip[1])}")

        for switch in topology['switches']:
            switchNode: "OVSKernelSwitch" = self.net.addSwitch(switch['id'])
            self.switches[switch["id"]] = switchNode
            switchNode.start([self.ryu])

        for link in topology['links']:
            self.net.addLink(
                self.net.get(link['source']), self.net.get(link['destination']), bw=link['bandwidth'])

        self.net.start()
        sleep(5)

        self.sdnController.assignSwitchIPs(topology, self.switches, hostGateways)


    def getNode(self, name: str) -> Host:
        """
        Get the node with the given name in the topology.

        Parameters:
        name (str): The name of the node to be returned.

        Returns:
        Host: The node with the given name in the topology.
        """

        return self.net.get(name)

    def getSwitches(self) -> Any:
        """
        Get the switches in the topology.
        """

        return self.switches

    def getHosts(self) -> Any:
        """
        Get the hosts in the topology.
        """

        return self.hosts

    def assignIPs(self, fg: ForwardingGraph) -> ForwardingGraph:
        """
        Assign IPs to the hosts in the topology.

        Parameters:
        fg (ForwardingGraph): The forwarding graph to be used to assign IPs.

        Returns:
        ForwardingGraph: The forwarding graph with the IPs assigned.
        """

        vnfs: VNF = fg['vnfs']
        vnfHosts: "TypedDict[str, Tuple[IPv4Address, IPv4Address]]" = {}

        def traverseVNF(vnfs: VNF, vnfHosts: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]"):
            """
            Traverse the VNFs in the forwarding graph and assigns IPs to the hosts.

            Parameters:
            vnfs (VNF): The VNF to be traversed.
            vnfHosts (TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]): The list of hosts in the VNF.
            """

            shouldContinue: bool = True

            while shouldContinue:
                if vnfs["host"]["id"] in vnfHosts:
                    vnfs["host"]["ip"] = vnfHosts[vnfs["host"]["id"]][2]
                else:
                    ipAddr: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()

                    vnfs['host']['ip'] = str(ipAddr[2])
                    vnfHosts[vnfs['host']['id']] = ipAddr

                    # Assign IP to the host
                    self.net.get(vnfs['host']['id']).cmd(
                        f"ip addr add {str(ipAddr[2])}/{ipAddr[0].prefixlen} dev {vnfs['host']['id']}-eth0")

                    self.sdnController.assignGatewayIP(self.topology, vnfs['host']['id'], ipAddr[1], self.switches)

                if isinstance(vnfs['next'], list):
                    for nextVnf in vnfs['next']:
                        traverseVNF(nextVnf, vnfHosts)

                    shouldContinue = False
                else:
                    vnfs = vnfs['next']

                if vnfs == "terminal":
                    shouldContinue = False

        traverseVNF(vnfs, vnfHosts)

        # Add gateways
        for name, ips in vnfHosts.items():
            host: Host = self.net.get(name)
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"ip route add {str(ips1[0])} via {ips[1]}")

        self.sdnController.installFlows(fg, vnfHosts, self.switches)
        CLI(self.net)
        self.net.stop()

        return fg
