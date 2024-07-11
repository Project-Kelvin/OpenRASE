"""
Defines the class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from threading import Thread
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
from constants.container import CPU_PERIOD, DIND_IMAGE, SERVER_IMAGE, SFCC_IMAGE, SFF_IMAGE, SFCC_CMD, SERVER_CMD, SFF_CMD
from mano.notification_system import NotificationSystem
from mano.sdn_controller import SDNController
from mano.telemetry import Telemetry
from utils.container import getContainerIP, waitTillContainerReady
from utils.embedding_graph import traverseVNF
from utils.tui import TUI


class InfraManager():
    """
    Class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
    """

    def __init__(self, sdnController: SDNController) -> None:
        """
        Constructor for the class.
        """

        self._topology: Topology = None
        self._networkIPs: "list[IPv4Network]" = []
        self._hosts: "dict[str, Host]" = {}
        self._switches: "dict[str, OVSKernelSwitch]" = {}
        self._sdnController: SDNController = None
        self._hostIPs: "dict[str, Tuple[IPv4Network, IPv4Address]]" = {}
        self._telemetry: Telemetry = None
        self._sfcHostIPs: "dict[str, dict[str, Tuple[IPv4Network, IPv4Address]]]" = {}
        self._sfcHostByIPs: "dict[str, tuple[str, str]]" = {}
        Telemetry.runSflow()
        self._sdnController = sdnController
        self._net: Any = Containernet()
        self._ryu: Ryu = Ryu('ryu', ryuArgs="ryu.app.ofctl_rest",
                       command="ryu-manager")
        self._net.addController(self._ryu)
        self._hostMACs: "dict[str, str]" = {}
        self._linkedPorts: "dict[str, tuple[int, int]]" = {}
        self._gatewayMACs: "dict[str, str]" = {}

    def installTopology(self, topology: Topology) -> None:
        """
        Spin up the provided topology virtually using Mininet (Containernet).

        Parameters:
            topology (Topology): The topology to be spun up.
        """

        try:
            TUI.appendToLog("Installing topology:")
            self._topology = topology
            self._telemetry = Telemetry(self._topology, self._sfcHostByIPs)

            config: Config = getConfig()

            # Add SFCC
            ipSFCC: "Tuple[IPv4Network, IPv4Address]" = generateIP(
                self._networkIPs)

            TUI.appendToLog("  Installing SFCC.")
            sfccDir: str = SFCC_IMAGE.split("/")[1].split(":")[0]
            sfcc: Host = self._net.addDocker(
                SFCC,
                ip=f"{ipSFCC[1]}/{ipSFCC[0].prefixlen}",
                dimage=SFCC_IMAGE,
                dcmd=SFCC_CMD,
                defaultRoute=f"dev {SFCC}-eth0",
                volumes=[
                    config["repoAbsolutePath"]
                    + f"/docker/files/{sfccDir}/shared/node-logs:/home/OpenRASE/apps/sfc_classifier/node-logs"
                ]
            )
            self._hosts[SFCC] = sfcc
            self._hostIPs[SFCC] = ipSFCC
            self._sfcHostByIPs[str(ipSFCC[1])] = (None, SFCC)

            # Add server
            ipServer: "Tuple[IPv4Network, IPv4Address]" = generateIP(
                self._networkIPs)
            self._hostIPs[SERVER] = ipServer

            TUI.appendToLog("  Installing server.")
            server: Host = self._net.addDocker(
                SERVER,
                ip=f"{ipServer[1]}/{ipServer[0].prefixlen}",
                dimage=SERVER_IMAGE,
                dcmd=SERVER_CMD,
                defaultRoute=f"dev {SERVER}-eth0"
            )
            self._hosts[SERVER] = server

            hostNodes: "list[Host]" = []

            TUI.appendToLog("  Installing hosts:")
            for host in topology['hosts']:
                TUI.appendToLog(f"    Installing host {host['id']}.")
                sffDir: str = SFF_IMAGE.split("/")[1].split(":")[0]
                sff: Host = self._net.addDocker(
                    host['id'],
                    ip=f"{getConfig()['sff']['network1']['sffIP']}/{getConfig()['sff']['network1']['mask']}",
                    dimage=SFF_IMAGE,
                    dcmd=SFF_CMD,
                    defaultRoute=f"dev {host['id']}-eth1",
                    volumes=[
                        config["repoAbsolutePath"]
                        + f"/docker/files/{sffDir}/shared/node-logs:/home/OpenRASE/apps/sff/node-logs"
                    ]
                )

                hostNode: Host = self._net.addDocker(
                    f"{host['id']}Node",
                    ip=f"{getConfig()['sff']['network1']['hostIP']}/{getConfig()['sff']['network1']['mask']}",
                    cpu_quota=int(host["cpu"] * CPU_PERIOD if "cpu" in host else -1),
                    mem_limit=f"{host['memory']}mb" if "memory" in host and host["memory"] is not None else None,
                    memswap_limit=f"{host['memory']}mb" if "memory" in host and host["memory"] is not None else None,
                    dimage=DIND_IMAGE,
                    privileged=True,
                    dcmd="dockerd",
                    volumes=[
                        config["repoAbsolutePath"]
                        + "/docker/files:/home/docker/files"
                    ]
                )
                hostNodes.append(hostNode)
                self._net.addLink(sff, hostNode)

                self._hosts[host["id"]] = sff

            TUI.appendToLog("  Installing switches:")
            for switch in topology['switches']:
                TUI.appendToLog(f"    Installing {switch['id']}.")
                switchNode: OVSKernelSwitch = self._net.addSwitch(switch['id'])
                TUI.appendToLog(f"    Installed {switch['id']} with id {switchNode.dpid}.")
                self._switches[switch["id"]] = switchNode
                switchNode.start([self._ryu])

            TUI.appendToLog("  Establishing links:")
            for link in topology['links']:
                TUI.appendToLog(f"    Linking {link['source']} to {link['destination']}.")
                mnLink: Any = self._net.addLink(
                    self._net.get(
                        link['source']),
                    self._net.get(link['destination']),
                    bw=link['bandwidth'] if 'bandwidth' in link else None)

                if link["source"] in self._hosts:
                    self._hostMACs[link["source"]] = mnLink.intf1.MAC()
                    self._gatewayMACs[link["source"]] = mnLink.intf2.MAC()
                elif link["destination"] in self._hosts:
                    self._hostMACs[link["destination"]] = mnLink.intf2.MAC()
                    self._gatewayMACs[link["destination"]] = mnLink.intf1.MAC()

                port1: int = mnLink.intf1.node.ports[mnLink.intf1]
                port2: int = mnLink.intf2.node.ports[mnLink.intf2]

                self._linkedPorts[f"{link['source']}-{link['destination']}"] = (port1, port2)
                self._linkedPorts[f"{link['destination']}-{link['source']}"] = (port2, port1)

            TUI.appendToLog("Starting Mininet.")
            try:
                self._net.start()
            except Exception as e:
                TUI.appendToLog(f"Non-critical error: {str(e)}", True)

            TUI.appendToLog("Waiting till Ryu is ready.")
            self._sdnController.waitTillReady(self._switches)

            TUI.appendToLog("Adding IP addresses to hosts.")
            for name, host in self._hosts.items():
                if name != SERVER and name != SFCC:
                    host.cmd(f"ip addr add {getConfig()['sff']['network2']['sffIP']}/{getConfig()['sff']['network2']['mask']} dev {name}-eth0")

            for hostNode in hostNodes:
                hostNode.cmd(
                    f"ip addr add {getConfig()['sff']['network2']['hostIP']}/{getConfig()['sff']['network2']['mask']} dev {hostNode.name}-eth0")

            TUI.appendToLog("Waiting till host containers are ready.")
            # Notify
            threads: "list[Thread]" = []
            for host in hostNodes:
                thread: Thread = Thread(target=waitTillContainerReady, args=(host.name,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            TUI.appendToLog("Topology installed successfully!")

            NotificationSystem.publish(TOPOLOGY_INSTALLED)
        except Exception as e:
            TUI.appendToLog(f"Error: {str(e)}", True)

    def getHostIPs(self) -> "dict[str, Tuple[IPv4Network, IPv4Address]]":
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

        TUI.appendToLog("  Deleting flows from switches:")

        try:
            self._sdnController.deleteFlows(
                eg, self._sfcHostIPs[eg["sfcID"]], self._switches, self._linkedPorts, self._hostMACs)
        except RuntimeError as e:
            TUI.appendToLog(f"  Error: {e}", True)

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
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]" = self._sfcHostIPs[eg["sfcID"]]
        else:
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]" = {
                SFCC: self._hostIPs[SFCC]
            }

        def traverseCallback(vnfs: VNF, _depth: int,
                             vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]") -> None:
            """
            Callback function for the traverseVNF function.

            Parameters:
                vnfs (VNF): The VNF.
                vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address]]):
                The hosts of the VNFs in the embedding graph.
            """

            if vnfs["host"]["id"] in vnfHosts:
                vnfs["host"]["ip"] = str(vnfHosts[vnfs["host"]["id"]][1])
            else:
                ipAddr: "Tuple[IPv4Network, IPv4Address]" = generateIP(
                    self._networkIPs)

                TUI.appendToLog(f"    Assigning IP {str(ipAddr[1])} to {vnfs['host']['id']}.")

                vnfs['host']['ip'] = str(ipAddr[1])
                vnfHosts[vnfs['host']['id']] = ipAddr
                self._sfcHostByIPs[str(ipAddr[1])] = (
                    eg["sfcID"], vnfs['host']['id'])

                # Assign IP to the host
                port: str = "eth0" if vnfs['host']['id'] == SERVER else "eth1"
                self._net.get(vnfs['host']['id']).cmd(
                    f"ip addr add {str(ipAddr[1])}/{ipAddr[0].prefixlen} dev {vnfs['host']['id']}-{port}")

                # Add ip to SFF
                hostIP: str = getContainerIP(vnfs["host"]["id"])
                TUI.appendToLog(f"    Adding host {vnfs['host']['id']} to SFF.")
                if not vnfs["next"] == TERMINAL:
                    try:
                        requests.post(f"http://{hostIP}:{getConfig()['sff']['port']}/add-host",
                                    json={"hostIP": str(ipAddr[1])},
                                    timeout=getConfig()["general"]["requestTimeout"])
                    except Exception as e:
                        TUI.appendToLog(f"    Error: {str(e)}", True)

        traverseVNF(vnfs, traverseCallback, vnfHosts)

        server: Host = self._net.get(SERVER)
        for name, ips in vnfHosts.items():
            host: Host = self._net.get(name)
            server.cmd(f"arp -s {str(ips[1])} {self._gatewayMACs[server.name]}")
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"arp -s {str(ips1[1])} {self._gatewayMACs[name]}")

        # Install flows
        TUI.appendToLog("    Installing flows in switches.")
        eg = self._sdnController.installFlows(eg, vnfHosts, self._switches, self._linkedPorts, self._hostMACs)
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
