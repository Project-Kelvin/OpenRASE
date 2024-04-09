"""
Defines the class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from time import sleep
from typing import Any, Tuple
import requests
from shared.constants.embedding_graph import TERMINAL
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.ip import generateIP
from shared.models.topology import Topology
from shared.models.embedding_graph import VNF, EmbeddingGraph
from mininet.node import Ryu, Host, OVSKernelSwitch
from mininet.net import Containernet
from mininet.cli import CLI
from constants.notification import TOPOLOGY_INSTALLED
from constants.topology import SERVER, SFCC
from constants.container import CPU_PERIOD, DIND_IMAGE, SERVER_IMAGE, SFCC_IMAGE, SFF_IMAGE
from mano.notification_system import NotificationSystem
from mano.sdn_controller import SDNController
from mano.telemetry import Telemetry
from utils.container import getContainerIP
from utils.embedding_graph import traverseVNF


class InfraManager():
    """
    Class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
    """

    _net: Any = None
    _ryu: Ryu = None
    _topology: Topology = None
    _networkIPs: "list[IPv4Network]" = []
    _hosts: "dict[str, Host]" = {}
    _switches: "dict[str, OVSKernelSwitch]" = {}
    _sdnController: SDNController = None
    _hostIPs: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = {}
    _telemetry: Telemetry = None
    _sfcHostIPs: "dict[str, dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]]" = {}
    _sfcHostByIPs: "dict[str, tuple[str, str]]" = {}

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
        self._telemetry = Telemetry(self._topology, self._sfcHostByIPs)

        config: Config = getConfig()

        # Add SFCC
        ipSFCC: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
            self._networkIPs)

        sfcc: Host = self._net.addDocker(
            SFCC,
            ip=f"{ipSFCC[2]}/{ipSFCC[0].prefixlen}",
            dimage=SFCC_IMAGE,
            dcmd="poetry run python sfc_classifier.py",
        )
        self._hosts[SFCC] = sfcc
        self._hostIPs[SFCC] = ipSFCC
        self._sfcHostByIPs[str(ipSFCC[2])] = (None, SFCC)

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
        self._hosts[SERVER] = server

        hostNodes: "list[Host]" = []

        for host in topology['hosts']:
            ip: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
                self._networkIPs)
            self._hostIPs[host["id"]] = ip

            sff: Host = self._net.addDocker(
                host['id'],
                ip=f"{getConfig()['sff']['network1']['sffIP']}/{getConfig()['sff']['network1']['mask']}",
                dimage=SFF_IMAGE,
                dcmd="poetry run python sff.py"
            )

            hostNode: Host = self._net.addDocker(
                f"{host['id']}Node",
                ip=f"{getConfig()['sff']['network1']['hostIP']}/{getConfig()['sff']['network1']['mask']}",
                cpu_quota=int(host["cpu"] * CPU_PERIOD if "cpu" in host else -1),
                mem_limit=f"{host['memory']}mb" if "memory" in host else None,
                memswap_limit=f"{host['memory']}mb" if "memory" in host else None,
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
            hostNodes.append(hostNode)
            self._net.addLink(sff, hostNode)
            #self._net.addLink(switch, hostNode)

            self._hosts[host["id"]] = sff

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

        for name, host in self._hosts.items():
            if name != SERVER and name != SFCC:
                host.cmd(f"ip addr add {str(self._hostIPs[name][2])}/{self._hostIPs[name][0].prefixlen} dev {name}-eth1")
                host.cmd(f"ip addr add {getConfig()['sff']['network2']['sffIP']}/{getConfig()['sff']['network2']['mask']} dev {name}-eth0")
            for name1, host1 in self._hosts.items():
                if host.name != host1.name:
                    host.cmd(
                        f"ip route add {str(self._hostIPs[name1][0])} via {str(self._hostIPs[name][1])}")

        for hostNode in hostNodes:
            hostNode.cmd(
                f"ip addr add {getConfig()['sff']['network2']['hostIP']}/{getConfig()['sff']['network2']['mask']} dev {hostNode.name}-eth0")

        self._sdnController.assignSwitchIPs(
            topology, self._switches, self._hostIPs, self._networkIPs)

        # Notify
        sleep(10)
        NotificationSystem.publish(TOPOLOGY_INSTALLED)

    def getHostIPs(self) -> "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]":
        """
        Get the IPs of the hosts in the topology.
        """

        return self._hostIPs

    def deleteSFC(self, eg: EmbeddingGraph) -> None:
        """
        Delete the SFC.

        Parameters:
            eg (EmbeddingGraph): The SFC to be deleted.
        """

        self._sdnController.deleteFlows(
            eg, self._sfcHostIPs[eg["sfcID"]], self._switches)

    def embedSFC(self, eg: EmbeddingGraph) -> EmbeddingGraph:
        """
        Assign IPs to the hosts in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to be used to assign IPs.

        Returns:
            EmbeddingGraph: The embedding graph with the IPs assigned.
        """

        vnfs: VNF = eg['vnfs']

        if eg["sfcID"] in self._sfcHostIPs:
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = self._sfcHostIPs[eg["sfcID"]]
        else:
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]" = {
                SFCC: self._hostIPs[SFCC]
            }

        def traverseCallback(vnfs: VNF, _depth: int,
                             vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]") -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnfs (VNF): The VNF.
                vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]):
                The hosts of the VNFs in the embedding graph.
            """

            if vnfs["host"]["id"] in vnfHosts:
                vnfs["host"]["ip"] = str(vnfHosts[vnfs["host"]["id"]][2])
            else:
                ipAddr: "Tuple[IPv4Network, IPv4Address, IPv4Address]" = generateIP(
                    self._networkIPs)

                vnfs['host']['ip'] = str(ipAddr[2])
                vnfHosts[vnfs['host']['id']] = ipAddr
                self._sfcHostByIPs[str(ipAddr[2])] = (
                    eg["sfcID"], vnfs['host']['id'])

                # Assign IP to the host
                port: str = "eth0" if vnfs['host']['id'] == SERVER else "eth1"
                self._net.get(vnfs['host']['id']).cmd(
                    f"ip addr add {str(ipAddr[2])}/{ipAddr[0].prefixlen} dev {vnfs['host']['id']}-{port}")

                self._sdnController.assignGatewayIP(
                    self._topology, vnfs['host']['id'], ipAddr[1], self._switches)

                # Add ip to SFF
                hostIP: str = getContainerIP(vnfs["host"]["id"])

                if not vnfs["next"] == TERMINAL:
                    requests.post(f"http://{hostIP}:{getConfig()['sff']['port']}/add-host",
                                  json={"hostIP": str(ipAddr[2])},
                                  timeout=getConfig()["general"]["requestTimeout"])

        traverseVNF(vnfs, traverseCallback, vnfHosts)

        # Add routes
        for name, ips in vnfHosts.items():
            host: Host = self._net.get(name)
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"ip route add {str(ips1[0])} via {ips[1]}")

        eg = self._sdnController.installFlows(eg, vnfHosts, self._switches)
        self._sfcHostIPs[eg["sfcID"]] = vnfHosts

        return eg

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
