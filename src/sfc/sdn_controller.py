"""
Defines the class that manipulates the Ryu SDN controller.
"""

from ipaddress import IPv4Address, IPv4Network
from typing import Tuple, TypedDict
from requests import Response
import requests
from shared.models.forwarding_graph import ForwardingLink
from shared.models.forwarding_graph import ForwardingGraph
from shared.models.topology import Link
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.models.topology import Topology
from utils.ryu import getRyuRestUrl


class SDNController():
    """
    Class that communicates with the Ryu SDN controller.
    """

    infraManager = None
    switchLinks: "TypedDict[str, IPv4Address]" = {}

    def __init__(self, infraManager) -> None:
        """
        Constructor for the class.
        """

        self.infraManager = infraManager

    def assignIP(self, ip: IPv4Address, switch: "OVSKernelSwitch") -> None:
        """
        Assign an IP address to a switch.

        Parameters:
        ip (str): The IP address to assign.
        switch (OVSKernelSwitch): The switch to assign the IP address to.

        Raises:
        RuntimeError: If the IP address could not be assigned.
        """
        config: Config = getConfig()

        data = {
            "address": f"{str(ip)}/{config['ipRange']['mask']}",
        }

        response: Response = requests.request(
            method="POST",
            url=getRyuRestUrl(switch.dpid),
            json=data,
            timeout=config["general"]["requestTimeout"]
        )

        if "failure" in str(response.content):
            raise RuntimeError(
                f"Failed to assign IP address {str(ip)} to switch {switch.name}.\n{response.json()}")

    def installFlow(self, destination: IPv4Network, gateway: IPv4Address, switch: "OVSKernelSwitch") -> None:
        """
        Install a flow in a switch.

        Parameters:
        destination (IPv4Network): The destination of the flow.
        gateway (IPv4Address): The gateway of the flow.
        switch (OVSKernelSwitch): The switch to install the flow in.

        Raises:
        RuntimeError: If the flow could not be installed.
        """

        config: Config = getConfig()

        data = {
            "gateway": str(gateway),
            "destination": str(destination),
        }

        response: Response = requests.request(
            method="POST",
            url=getRyuRestUrl(switch.dpid),
            json=data,
            timeout=config["general"]["requestTimeout"]
        )

        if "failure" in str(response.content):
            if "Destination overlaps" not in str(response.content):
                raise RuntimeError(
                    f"Failed to install flow in switch {switch.name}.\n{response.json()}")


    def assignSwitchIPs(self, topology: Topology, switches: "TypedDict[str, OVSKernelSwitch]",
                        hostIPs: "TypedDict[str, (IPv4Network, IPv4Address, IPv4Address)]") -> None:
        """
        Assign IP addresses to the switches in the topology.

        Parameters:
        topology (Topology): The topology to assign IP addresses to.
        switches ("list[OVSKernelSwitch]"): The switches to assign IP addresses to.
        hostIPs ("TypedDict[str, (IPv4Network, IPv4Address, IPv4Address)]"):
        The gateways of the hosts in the topology.
        """

        links: "list[Link]" = topology["links"]

        for link in links:
            if link["source"] in switches and link["destination"] in switches:
                _networkAddr, addr1, addr2 = self.infraManager.generateIP()
                self.assignIP(addr1, switches[link["source"]])
                self.assignIP(addr2, switches[link["destination"]])
                self.switchLinks[f'{link["source"]}-{link["destination"]}'] = addr1
                self.switchLinks[f'{link["destination"]}-{link["source"]}'] = addr2
            else:
                if link["source"] in switches:
                    self.assignIP(hostIPs[link["destination"]][1], switches[link["source"]])
                    self.switchLinks[f'{link["source"]}-{link["destination"]}'] = hostIPs[link["destination"]][1]
                elif link["destination"] in switches:
                    self.assignIP(hostIPs[link["source"]][1], switches[link["destination"]])
                    self.switchLinks[f'{link["source"]}-{link["destination"]}'] = hostIPs[link["source"]][1]

    def assignGatewayIP(self, topology: Topology, host: str, ip: IPv4Address,
                        switches: "TypedDict[str, OVSKernelSwitch]") -> None:
        """
        Assign IP addresses to the gateways of the hosts in the topology.

        Parameters:
        topology (Topology): The topology to assign IP addresses to.
        host (str): The host to assign the gateway IP address to.
        ip (IPv4Address): The IP address to assign.
        switches ("TypedDict[str, OVSKernelSwitch]"): The switches to assign IP addresses to.
        """

        links: "list[Link]" = topology["links"]

        for link in links:
            if link["source"] == host:
                self.assignIP(ip, switches[link["destination"]])
            elif link["destination"] == host:
                self.assignIP(ip, switches[link["source"]])

    def installFlows(self, fg: ForwardingGraph,
                     vnfHosts: "TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]",
                     switches: "TypedDict[str, OVSKernelSwitch]") -> ForwardingGraph:
        """
        Install flows in the switches in the topology.

        Parameters:
        fg (ForwardingGraph): The forwarding graph to install flows in.
        vnfHosts (TypedDict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]):
        The hosts of the VNFs in the forwarding graph.
        switches ("TypedDict[str, OVSKernelSwitch]"): The switches to install flows in.

        Returns:
        ForwardingGraph: The forwarding graph with the flows installed.
        """

        links: "list[ForwardingLink]" = fg["links"]

        for link in links:
            sourceNetwork: IPv4Network = vnfHosts[link["source"]["id"]][0]
            link["source"]["ip"] = str(vnfHosts[link["source"]["id"]][2])
            destinationNetwork: IPv4Network = vnfHosts[link["destination"]["id"]][0]
            link["destination"]["ip"] = str(vnfHosts[link["destination"]["id"]][2])

            for index, switch in enumerate(link["links"]):
                nextSwitch: str = link["links"][index + 1] if index < len(link["links"]) - 1 else None
                prevSwitch: str = link["links"][index - 1] if index > 0 else None

                if nextSwitch is not None:
                    self.installFlow(destinationNetwork, self.switchLinks[f"{nextSwitch}-{switch}"],
                                     switches[switch])
                if prevSwitch is not None:
                    self.installFlow(sourceNetwork, self.switchLinks[f"{prevSwitch}-{switch}"],
                                     switches[switch])

        return fg
