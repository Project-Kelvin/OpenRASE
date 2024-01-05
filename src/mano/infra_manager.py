"""
Defines the class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from time import sleep
from typing import Any, Tuple, TypedDict
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.ip import generateIP
from shared.models.topology import Topology
from shared.models.forwarding_graph import VNF, ForwardingGraph
from mininet.node import Ryu, Host, OVSKernelSwitch
from mininet.net import Containernet
from mininet.cli import CLI
from constants.notification import TOPOLOGY_INSTALLED
from constants.topology import SERVER, SFCC, SFCC_SWITCH, TRAFFIC_GENERATOR
from constants.container import CPU_PERIOD, DIND_IMAGE, SERVER_IMAGE, SFCC_IMAGE
from mano.notification_system import NotificationSystem
from mano.sdn_controller import SDNController
from mano.telemetry import Telemetry
from utils.forwarding_graph import traverseVNF

class InfraManager():
    """
    Class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
    """

    _net: Any = None
    _ryu: Ryu = None
    _topology: Topology = None
    _networkIPs: "list[IPv4Network]" = []
    _hosts: "TypedDict[str, Host]" = {}
    _switches: "TypedDict[str, OVSKernelSwitch]" = {}
    _sdnController: SDNController = None
    _hostIPs: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = {}
    _telemetry: Telemetry = None

    def __init__(self, sdnController: SDNController) -> None:
        """
        Constructor for the class.
        """

        Telemetry.runSflow()
        self._sdnController = sdnController
        self._net = Containernet()
        self._ryu = Ryu('ryu', ryuArgs="ryu.app.rest_router",
                       command="ryu-manager")
        self._net.addController(self._ryu)

    def installTopology(self, topology: Topology) -> None:
        """
        Spin up the provided topology virtually using Mininet (Containernet).

        Parameters:
            topology (Topology): The topology to be spun up.
        """

        self._topology = topology
        self._telemetry = Telemetry(self._topology)

        config: Config = getConfig()

        # Add traffic generator
        ipTG: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(self._networkIPs)

        tg: Host = self._net.addDocker(
            TRAFFIC_GENERATOR,
            ip=f"{ipTG[2]}/{ipTG[0].prefixlen}",
            dimage=DIND_IMAGE,
            privileged=True,
            dcmd="dockerd",
            volumes=[
                config["repoAbsolutePath"]
                + "/docker/compose:/home/docker"
            ]
        )

        # Add SFCC Switch
        sfccTGSwitch: OVSKernelSwitch = self._net.addSwitch(SFCC_SWITCH)
        sfccTGSwitch.start([self._ryu])

        # Add SFCC Switch-TG Link
        self._net.addLink(tg, sfccTGSwitch)

        # Add SFCC
        ipSFCCTG: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
            self._networkIPs)

        sfcc: Host = self._net.addDocker(
            SFCC,
            ip=f"{ipSFCCTG[2]}/{ipSFCCTG[0].prefixlen}",
            dimage=SFCC_IMAGE,
            dcmd="poetry run python sfc_classifier.py",
        )
        self._hosts[SFCC]=sfcc

        # Add SFCC Link
        self._net.addLink(sfccTGSwitch, sfcc)

        # Add server
        ipServer: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
            self._networkIPs)
        self._hostIPs[SERVER] = ipServer

        server: Host = self._net.addDocker(
            SERVER,
            ip=f"{ipServer[2]}/{ipServer[0].prefixlen}",
            dimage=SERVER_IMAGE,
            dcmd="poetry run python server.py"
        )
        self._hosts[SERVER]=server

        for host in topology['hosts']:
            ip: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
                self._networkIPs)
            self._hostIPs[host["id"]] = ip

            hostNode: Host = self._net.addDocker(
                host['id'],
                ip=f"{str(ip[2])}/{ip[0].prefixlen}",
                cpu_quota=host['cpu'] * CPU_PERIOD,
                mem_limit=host['memory'],
                memswap_limit=host['memory'],
                dimage=DIND_IMAGE,
                privileged=True,
                dcmd="dockerd",
                volumes=[
                    config["repoAbsolutePath"]
                    + "/docker/files:/home/docker/files",
                    config["repoAbsolutePath"]
                    + "/docker/compose:/home/docker/compose"
                ]
            )
            self._hosts[host["id"]]=hostNode

        for switch in topology['switches']:
            switchNode: OVSKernelSwitch = self._net.addSwitch(switch['id'])
            self._switches[switch["id"]] = switchNode
            switchNode.start([self._ryu])

        for link in topology['links']:
            self._net.addLink(
                self._net.get(
                    link['source']),
                    self._net.get(link['destination']),
                    bw=link['bandwidth'] if 'bandwidth' in link else None)

        self._net.start()
        sleep(5)

        # Add ip to SFCC's interface connecting to the switch that connects to the rest of the topology.
        ipSFCCTopo: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
            self._networkIPs)
        sfcc.setIP(str(ipSFCCTopo[2]), prefixLen=ipSFCCTopo[0].prefixlen, intf=f"{sfcc.name}-eth1")
        self._hostIPs[SFCC] = ipSFCCTopo

        # Add routes
        sfcc.cmd(f"ip route add {str(ipTG[0])} via {str(ipSFCCTG[1])}")
        tg.cmd(f"ip route add {str(ipSFCCTG[0])} via {str(ipTG[1])}")

        for name, host in self._hosts.items():
            for name1, host1 in self._hosts.items():
                if host.name != host1.name:
                    host.cmd(
                        f"ip route add {str(self._hostIPs[name1][0])} via {str(self._hostIPs[name][1])}")

        # Add SFCC switch ip addresses
        self._sdnController.assignIP(ipSFCCTG[1], sfccTGSwitch)
        self._sdnController.assignIP(ipTG[1], sfccTGSwitch)

        self._sdnController.assignSwitchIPs(topology, self._switches, self._hostIPs, self._networkIPs)

        # Notify
        sleep(10)
        NotificationSystem.publish(TOPOLOGY_INSTALLED)

    def getHostIPs(self) -> "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]":
        """
        Get the IPs of the hosts in the topology.
        """

        return self._hostIPs

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
            SFCC: self._hostIPs[SFCC]
        }

        def traverseCallback(vnfs: VNF,
                             vnfHosts: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]") -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnfs (VNF): The VNF.
                vnfHosts (TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]):
                The hosts of the VNFs in the forwarding graph.
            """

            if vnfs["host"]["id"] in vnfHosts:
                vnfs["host"]["ip"] = vnfHosts[vnfs["host"]["id"]][2]
            else:
                ipAddr: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
                    self._networkIPs)

                vnfs['host']['ip'] = str(ipAddr[2])
                vnfHosts[vnfs['host']['id']] = ipAddr

                # Assign IP to the host
                self._net.get(vnfs['host']['id']).cmd(
                    f"ip addr add {str(ipAddr[2])}/{ipAddr[0].prefixlen} dev {vnfs['host']['id']}-eth0")

                self._sdnController.assignGatewayIP(self._topology, vnfs['host']['id'], ipAddr[1], self._switches)

        traverseVNF(vnfs, traverseCallback, vnfHosts)

        # Add routes
        for name, ips in vnfHosts.items():
            host: Host = self._net.get(name)
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"ip route add {str(ips1[0])} via {ips[1]}")

        fg = self._sdnController.installFlows(fg, vnfHosts, self._switches)

        return fg

    def stopNetwork(self) -> None:
        """
        Stop the network.
        """

        self._net.stop()

    def startCLI(self) -> None:
        """
        Start the CLI.
        """

        CLI(self._net)

    def getTopology(self) -> Topology:
        """
        Get the topology.
        """

        return self._topology

    def getTelemetry(self) -> Telemetry:
        """
        Get the telemetry.
        """

        return self._telemetry
