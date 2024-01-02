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
from constants.topology import SERVER, SFCC, SFCC_SWITCH, TRAFFIC_GENERATOR
from constants.docker import DIND, SERVER_IMAGE

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
    hostIPs: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = {}

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

        # Add traffic generator
        ipTG: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()

        tg: Host = self.net.addDocker(
            TRAFFIC_GENERATOR,
            ip=f"{ipTG[2]}/{ipTG[0].prefixlen}",
            dimage=DIND,
            privileged=True,
            dcmd="dockerd",
            volumes=[
                config["repoAbsolutePath"]
                + "/docker/compose:/home/docker"
            ]
        )

        # Add SFCC Switch
        sfccTGSwitch: "OVSKernelSwitch" = self.net.addSwitch(SFCC_SWITCH)
        sfccTGSwitch.start([self.ryu])

        # Add SFCC Switch-TG Link
        self.net.addLink(tg, sfccTGSwitch)

        # Add SFCC
        ipSFCCTG: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()

        sfcc: Host = self.net.addDocker(
            SFCC,
            ip=f"{ipSFCCTG[2]}/{ipSFCCTG[0].prefixlen}",
            dimage=DIND,
            privileged=True,
            dcmd="dockerd",
            volumes=[
                config["repoAbsolutePath"]
                + "/docker/compose:/home/docker"
            ]
        )
        self.hosts[SFCC]=sfcc

        # Add SFCC Link
        self.net.addLink(sfccTGSwitch, sfcc)

        # Add server
        ipServer: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()
        self.hostIPs[SERVER] = ipServer

        server: Host = self.net.addDocker(
            SERVER,
            ip=f"{ipServer[2]}/{ipServer[0].prefixlen}",
            dimage=SERVER_IMAGE,
            privileged=True,
            dcmd="poetry run python server.py"
        )
        self.hosts[SERVER]=server

        for host in topology['hosts']:
            ip: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()
            self.hostIPs[host["id"]] = ip

            hostNode: Host = self.net.addDocker(
                host['id'],
                ip=f"{str(ip[2])}/{ip[0].prefixlen}",
                cpu_quota=host['cpu'] * cpuPeriod,
                mem_limit=host['memory'],
                memswap_limit=host['memory'],
                dimage=DIND,
                privileged=True,
                dcmd="dockerd",
                volumes=[
                    config["repoAbsolutePath"]
                    + "/docker/files:/home/docker"
                ]
            )
            self.hosts[host["id"]]=hostNode

        for switch in topology['switches']:
            switchNode: "OVSKernelSwitch" = self.net.addSwitch(switch['id'])
            self.switches[switch["id"]] = switchNode
            switchNode.start([self.ryu])

        for link in topology['links']:
            self.net.addLink(
                self.net.get(link['source']), self.net.get(link['destination']), bw=link['bandwidth'] if 'bandwidth' in link else None)

        self.net.start()
        sleep(5)

        # Add ip to SFCC's interface connecting to the switch that connects to the rest of teh topology.
        ipSFCCTopo: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = self.generateIP()
        sfcc.setIP(str(ipSFCCTopo[2]), prefixLen=ipSFCCTopo[0].prefixlen, intf=f"{sfcc.name}-eth1")
        self.hostIPs[SFCC] = ipSFCCTopo

        # Add routes
        sfcc.cmd(f"ip route add {str(ipTG[0])} via {str(ipSFCCTG[1])}")
        tg.cmd(f"ip route add {str(ipSFCCTG[0])} via {str(ipTG[1])}")

        for host in self.hosts:
            for host1 in self.hosts:
                if self.hosts[host].name != self.hosts[host1].name:
                    self.hosts[host].cmd(
                        f"ip route add {str(self.hostIPs[host1][0])} via {str(self.hostIPs[host][1])}")

        # Add SFCC switch ip addresses
        self.sdnController.assignIP(ipSFCCTG[1], sfccTGSwitch)
        self.sdnController.assignIP(ipTG[1], sfccTGSwitch)

        self.sdnController.assignSwitchIPs(topology, self.switches, self.hostIPs)


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
        vnfHosts: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = {
            SFCC: self.hostIPs[SFCC]
        }

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

        # Add routes
        for name, ips in vnfHosts.items():
            host: Host = self.net.get(name)
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"ip route add {str(ips1[0])} via {ips[1]}")

        fg = self.sdnController.installFlows(fg, vnfHosts, self.switches)

        CLI(self.net)
        self.net.stop()

        return fg
