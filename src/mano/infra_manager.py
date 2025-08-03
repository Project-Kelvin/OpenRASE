"""
Defines the class that corresponds to the Virtualized Infrastructure Manager in the NFV architecture.
"""

from ipaddress import IPv4Address, IPv4Network
from threading import Thread
from typing import Any, Tuple
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.ip import generateIP
from shared.models.topology import Topology
from shared.models.embedding_graph import VNF, EmbeddingGraph
from mininet.node import Ryu, Host, OVSKernelSwitch
from mininet.net import Containernet
from mininet.cli import CLI
from mininet.link import TCLink
from constants.notification import TOPOLOGY_INSTALLED
from constants.topology import SERVER, SFCC
from constants.container import (
    SERVER_IMAGE,
    SFCC_IMAGE,
    SFCC_CMD,
    SERVER_CMD,
)
from utils.liveness_checker import checkLiveness
from utils.container import waitTillContainerReady
from utils.embedding_graph import traverseVNF
from utils.host import addHostNode, addSFF, addSFFEnds
from utils.tui import TUI
from mano.notification_system import NotificationSystem
from mano.sdn_controller import SDNController
from mano.telemetry import Telemetry


class InfraManager:
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
        self._ryu: Ryu = Ryu("ryu", ryuArgs="ryu.app.ofctl_rest", command="ryu-manager")
        self._net.addController(self._ryu)
        self._hostMACs: "dict[str, str]" = {}
        self._linkedPorts: "dict[str, tuple[int, int]]" = {}
        self._gatewayMACs: "dict[str, str]" = {}
        self._stopLivenessChecker: "list[bool]" = [False]
        self._firstHostIPs: "dict[str, Tuple[IPv4Network, IPv4Address]]" = {}

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
            ipSFCC: "Tuple[IPv4Network, IPv4Address]" = generateIP(self._networkIPs)

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
                ],
            )
            self._hosts[SFCC] = sfcc
            self._hostIPs[SFCC] = ipSFCC
            self._sfcHostByIPs[str(ipSFCC[1])] = (None, SFCC)
            self._firstHostIPs[SFCC] = ipSFCC

            # Add server
            ipServer: "Tuple[IPv4Network, IPv4Address]" = generateIP(self._networkIPs)
            self._hostIPs[SERVER] = ipServer

            TUI.appendToLog("  Installing server.")
            server: Host = self._net.addDocker(
                SERVER,
                ip=f"{ipServer[1]}/{ipServer[0].prefixlen}",
                dimage=SERVER_IMAGE,
                dcmd=SERVER_CMD,
                defaultRoute=f"dev {SERVER}-eth0",
            )
            self._hosts[SERVER] = server

            hostNodes: "list[Host]" = []
            sffs: "list[Host]" = []

            TUI.appendToLog("  Installing hosts:")
            for host in topology["hosts"]:
                TUI.appendToLog(f"    Installing host {host['id']}.")
                sff: Host = addSFF(host, self._net)
                hostNode: Host = addHostNode(host, self._net)
                hostNodes.append(hostNode)
                sffs.append(sff)
                self._net.addLink(sff, hostNode)

                self._hosts[host["id"]] = sff

            TUI.appendToLog("  Installing switches:")
            for switch in topology["switches"]:
                TUI.appendToLog(f"    Installing {switch['id']}.")
                switchNode: OVSKernelSwitch = self._net.addSwitch(switch["id"])
                TUI.appendToLog(
                    f"    Installed {switch['id']} with id {switchNode.dpid}."
                )
                self._switches[switch["id"]] = switchNode
                switchNode.start([self._ryu])

            TUI.appendToLog("  Establishing links:")
            for link in topology["links"]:
                TUI.appendToLog(
                    f"    Linking {link['source']} to {link['destination']}."
                )
                mnLink: Any = self._net.addLink(
                    self._net.get(link["source"]),
                    self._net.get(link["destination"]),
                    bw=link["bandwidth"] if "bandwidth" in link else None,
                    delay=f"{link['delay']}ms" if "delay" in link else None,
                    cls=TCLink,
                )

                if link["source"] in self._hosts:
                    self._hostMACs[link["source"]] = mnLink.intf1.MAC()
                    self._gatewayMACs[link["source"]] = mnLink.intf2.MAC()
                elif link["destination"] in self._hosts:
                    self._hostMACs[link["destination"]] = mnLink.intf2.MAC()
                    self._gatewayMACs[link["destination"]] = mnLink.intf1.MAC()

                port1: int = mnLink.intf1.node.ports[mnLink.intf1]
                port2: int = mnLink.intf2.node.ports[mnLink.intf2]

                self._linkedPorts[f"{link['source']}-{link['destination']}"] = (
                    port1,
                    port2,
                )
                self._linkedPorts[f"{link['destination']}-{link['source']}"] = (
                    port2,
                    port1,
                )

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
                    host.cmd(
                        f"ip addr add {getConfig()['sff']['network2']['sffIP']}/{getConfig()['sff']['network2']['mask']} dev {name}-eth0"
                    )

            for hostNode in hostNodes:
                hostNode.cmd(
                    f"ip addr add {getConfig()['sff']['network2']['hostIP']}/{getConfig()['sff']['network2']['mask']} dev {hostNode.name}-eth0"
                )

            TUI.appendToLog("Waiting till host containers are ready.")
            threads: "list[Thread]" = []
            for host in hostNodes:
                TUI.appendToLog(f"  Waiting for {host.name} to be ready.")
                thread: Thread = Thread(
                    target=waitTillContainerReady, args=(host.name,)
                )
                thread.start()
                threads.append(thread)

            for sff in sffs:
                TUI.appendToLog(f"  Waiting for {sff} to be ready.")
                thread: Thread = Thread(target=waitTillContainerReady, args=(sff.name,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            TUI.appendToLog("Initiating RX and TX ends of SFF.")

            sffThreads: "list[Thread]" = []
            for sff in sffs:
                thread: Thread = Thread(target=addSFFEnds, args=(sff.name,))
                thread.start()
                sffThreads.append(thread)

            for thread in sffThreads:
                thread.join()

            TUI.appendToLog("Topology installed successfully!")
            checkLiveness(self._net, self._topology, self._stopLivenessChecker)
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
            self._sdnController.configureSwitchFlows(
                eg,
                self._sfcHostIPs[eg["sfcID"]],
                self._switches,
                self._linkedPorts,
                self._hostMACs,
                self._firstHostIPs,
                False,
            )
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

        vnfs: VNF = eg["vnfs"]

        if eg["sfcID"] in self._sfcHostIPs:
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]" = self._sfcHostIPs[
                eg["sfcID"]
            ]
        else:
            sfccIP: "Tuple[IPv4Network, IPv4Address]" = generateIP(self._networkIPs)

            TUI.appendToLog(f"    Assigning IP {str(sfccIP[1])} to {SFCC}.")

            self._sfcHostByIPs[str(sfccIP[1])] = (eg["sfcID"], SFCC)
            self._net.get(SFCC).cmd(
                f"ip addr add {str(sfccIP[1])}/{sfccIP[0].prefixlen} dev {SFCC}-eth0"
            )

            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]" = {SFCC: sfccIP}

        def traverseCallback(
            vnfs: VNF,
            _depth: int,
            vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]",
        ) -> None:
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
                ipAddr: "Tuple[IPv4Network, IPv4Address]" = generateIP(self._networkIPs)

                TUI.appendToLog(
                    f"    Assigning IP {str(ipAddr[1])} to {vnfs['host']['id']}."
                )

                vnfs["host"]["ip"] = str(ipAddr[1])

                if vnfs["host"]["id"] not in self._firstHostIPs:
                    self._firstHostIPs[vnfs["host"]["id"]] = ipAddr

                vnfHosts[vnfs["host"]["id"]] = ipAddr
                self._sfcHostByIPs[str(ipAddr[1])] = (eg["sfcID"], vnfs["host"]["id"])

                # Assign IP to the host
                port: str = "eth0" if vnfs["host"]["id"] == SERVER else "eth1"
                self._net.get(vnfs["host"]["id"]).cmd(
                    f"ip addr add {str(ipAddr[1])}/{ipAddr[0].prefixlen} dev {vnfs['host']['id']}-{port}"
                )

        traverseVNF(vnfs, traverseCallback, vnfHosts)

        for name in vnfHosts.keys():
            host: Host = self._net.get(name)
            for name1, ips1 in vnfHosts.items():
                if name != name1:
                    host.cmd(f"arp -s {str(ips1[1])} {self._gatewayMACs[name]}")

        # Install flows
        TUI.appendToLog("    Installing flows in switches.")
        eg = self._sdnController.configureSwitchFlows(
            eg,
            vnfHosts,
            self._switches,
            self._linkedPorts,
            self._hostMACs,
            self._firstHostIPs,
            True,
        )
        self._sfcHostIPs[eg["sfcID"]] = vnfHosts

        return eg

    def stopNetwork(self) -> None:
        """
        Stop the network.
        """

        self._stopLivenessChecker[0] = True
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
