"""
Defines the class that manipulates the Ryu SDN controller.
"""

from ipaddress import IPv4Address, IPv4Network
from typing import TypedDict
from requests import Response
import requests
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
        config: Config = getConfig()

        response: Response = requests.request(
            method="POST",
            url=getRyuRestUrl(switch.dpid),
            json=data,
            timeout=config["general"]["requestTimeout"]
        )

        if "failure" in str(response.content):
            raise RuntimeError(
                f"Failed to assign IP address {ip} to switch {switch.name}.\n{response.json()['details']}")

    def assignSwitchIPs(self, topology: Topology, switches: "TypedDict[str, OVSKernelSwitch]",
                        hostGateways: "TypedDict[str, IPv4Address]") -> None:
        """
        Assign IP addresses to the switches in the topology.

        Parameters:
        topology (Topology): The topology to assign IP addresses to.
        switches ("list[OVSKernelSwitch]"): The switches to assign IP addresses to.
        hostGateways ("TypedDict[str, IPv4Address]"): The gateways of the hosts in the topology.
        """

        links: "list[Link]" = topology["links"]

        for link in links:
            if link["source"] in switches and link["destination"] in switches:
                _networkAddr, addr1, addr2 = self.infraManager.generateIP()
                self.assignIP(addr1, switches[link["source"]])
                self.assignIP(addr2, switches[link["destination"]])
            else:
                if link["source"] in switches:
                    self.assignIP(hostGateways[link["destination"]], switches[link["source"]])
                elif link["destination"] in switches:
                    self.assignIP(hostGateways[link["source"]], switches[link["destination"]])

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
