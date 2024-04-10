"""
Defines the class that manipulates the Ryu SDN controller.
"""

from ipaddress import IPv4Address, IPv4Network
from threading import Thread
from typing import Tuple
from requests import Response
import requests
from shared.models.embedding_graph import EmbeddingGraph, ForwardingLink
from shared.models.topology import Link
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.models.topology import Topology
from shared.utils.ip import generateIP
from mininet.node import OVSKernelSwitch
from utils.ryu import getRyuRestUrl

class SDNController():
    """
    Class that communicates with the Ryu SDN controller.
    """

    _switchLinks: "dict[str, IPv4Address]" = {}

    def assignIP(self, ip: IPv4Address, switch: OVSKernelSwitch) -> None:
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

    def _installFlow(self, destination: IPv4Network, gateway: IPv4Address, switch: OVSKernelSwitch) -> None:
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

    def _deleteFlow(self, destination: IPv4Network, gateway: IPv4Address, switch: OVSKernelSwitch) -> None:
        """
        Delete a flow in a switch.

        Parameters:
            destination (IPv4Network): The destination of the flow.
            gateway (IPv4Address): The gateway of the flow.
            switch (OVSKernelSwitch): The switch to delete the flow from.

        Raises:
            RuntimeError: If the flow could not be deleted.
        """

        response: Response = requests.request(
            method="GET",
            url=getRyuRestUrl(switch.dpid),
            timeout=getConfig()["general"]["requestTimeout"]
        )

        if "failure" in str(response.content):
            raise RuntimeError(
                f"Failed to delete flow in switch {switch.name}.\n{response.json()}")
        else:
            flows: "list[dict]" = response.json()[0]["internal_network"][0]["route"]

            for flow in flows:
                if flow["destination"] == str(destination) and flow["gateway"] == str(gateway):
                    data = {
                        "route_id": flow["route_id"]
                    }
                    response = requests.request(
                        method="DELETE",
                        url=getRyuRestUrl(switch.dpid),
                        json=data,
                        timeout=getConfig()["general"]["requestTimeout"]
                    )

                    if "failure" in str(response.content):
                        raise RuntimeError(
                            f"Failed to delete flow in switch {switch.name}.\n{response.json()}")
                    break

    def assignSwitchIPs(self, topology: Topology, switches: "dict[str, OVSKernelSwitch]",
                        hostIPs: "dict[str, (IPv4Network, IPv4Address, IPv4Address)]",
                        existingIPs: "list[IPv4Network]") -> None:
        """
        Assign IP addresses to the switches in the topology.

        Parameters:
            topology (Topology): The topology to assign IP addresses to.
            switches (dict[str, OVSKernelSwitch]): The switches to assign IP addresses to.
            hostIPs (dict[str, (IPv4Network, IPv4Address, IPv4Address)]):
                The gateways of the hosts in the topology.
            existingIPs (list[IPv4Network]): The IP addresses that are already in use.
        """

        links: "list[Link]" = topology["links"]

        for link in links:
            if link["source"] in switches and link["destination"] in switches:
                _networkAddr, addr1, addr2 = generateIP(existingIPs)
                self.assignIP(addr1, switches[link["source"]])
                self.assignIP(addr2, switches[link["destination"]])
                self._switchLinks[f'{link["source"]}-{link["destination"]}'] = addr1
                self._switchLinks[f'{link["destination"]}-{link["source"]}'] = addr2
            else:
                if link["source"] in switches:
                    self.assignIP(hostIPs[link["destination"]][1], switches[link["source"]])
                    self._switchLinks[f'{link["source"]}-{link["destination"]}'] = hostIPs[link["destination"]][1]
                elif link["destination"] in switches:
                    self.assignIP(hostIPs[link["source"]][1], switches[link["destination"]])
                    self._switchLinks[f'{link["source"]}-{link["destination"]}'] = hostIPs[link["source"]][1]

    def assignGatewayIP(self, topology: Topology, host: str, ip: IPv4Address,
                        switches: "dict[str, OVSKernelSwitch]") -> None:
        """
        Assign IP addresses to the gateways of the hosts in the topology.

        Parameters:
            topology (Topology): The topology to assign IP addresses to.
            host (str): The host to assign the gateway IP address to.
            ip (IPv4Address): The IP address to assign.
            switches (dict[str, OVSKernelSwitch]): The switches to assign IP addresses to.
        """

        links: "list[Link]" = topology["links"]

        for link in links:
            if link["source"] == host:
                self.assignIP(ip, switches[link["destination"]])
            elif link["destination"] == host:
                self.assignIP(ip, switches[link["source"]])

    def installFlows(self, eg: EmbeddingGraph,
                     vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]",
                     switches: "dict[str, OVSKernelSwitch]") -> EmbeddingGraph:
        """
        Install flows in the switches in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to install flows in.
            vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]):
            The hosts of the VNFs in the embedding graph.
            switches (dict[str, OVSKernelSwitch]"): The switches to install flows in.

        Returns:
            EmbeddingGraph: The embedding graph with the flows installed.
        """

        links: "list[ForwardingLink]" = eg["links"]

        for link in links:
            sourceNetwork: IPv4Network = vnfHosts[link["source"]["id"]][0]
            link["source"]["ip"] = str(vnfHosts[link["source"]["id"]][2])
            destinationNetwork: IPv4Network = vnfHosts[link["destination"]["id"]][0]
            link["destination"]["ip"] = str(vnfHosts[link["destination"]["id"]][2])

            for index, switch in enumerate(link["links"]):
                nextSwitch: str = link["links"][index + 1] if index < len(link["links"]) - 1 else None
                prevSwitch: str = link["links"][index - 1] if index > 0 else None

                if nextSwitch is not None:
                    self._installFlow(destinationNetwork, self._switchLinks[f"{nextSwitch}-{switch}"],
                                     switches[switch])
                if prevSwitch is not None:
                    self._installFlow(sourceNetwork, self._switchLinks[f"{prevSwitch}-{switch}"],
                                     switches[switch])

        return eg

    def deleteFlows(self, eg: EmbeddingGraph,
                     vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]",
                     switches: "dict[str, OVSKernelSwitch]") -> None:
        """
        Delete flows in the switches in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to delete flows from.
            vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address, IPv4Address]]):
            The hosts of the VNFs in the embedding graph.
            switches (dict[str, OVSKernelSwitch]"): The switches to delete flows from.
        """

        links: "list[ForwardingLink]" = eg["links"]

        for link in links:
            sourceNetwork: IPv4Network = vnfHosts[link["source"]["id"]][0]
            destinationNetwork: IPv4Network = vnfHosts[link["destination"]["id"]][0]

            for index, switch in enumerate(link["links"]):
                nextSwitch: str = link["links"][index +
                                                1] if index < len(link["links"]) - 1 else None
                prevSwitch: str = link["links"][index -
                                                1] if index > 0 else None

                if nextSwitch is not None:
                    self._deleteFlow(destinationNetwork, self._switchLinks[f"{nextSwitch}-{switch}"],
                                      switches[switch])
                if prevSwitch is not None:
                    self._deleteFlow(sourceNetwork, self._switchLinks[f"{prevSwitch}-{switch}"],
                                      switches[switch])

    def waitTillReady(self, switches: "dict[str, OVSKernelSwitch]" ) -> None:
        """
        Wait until the SDN controller is ready.

        Parameters:
            switches (dict[str, OVSKernelSwitch]): The switches to wait for.
        """

        config: Config = getConfig()

        threads: "list[Thread]" = []

        def checkSwitch(switch: OVSKernelSwitch):
            isReady: bool = False
            while not isReady:
                try:
                    response: Response = requests.request(
                        method="GET",
                        url=getRyuRestUrl(switch.dpid),
                        timeout=config["general"]["requestTimeout"]
                    )
                    if "failure" not in str(response.content):
                        isReady = True
                # pylint: disable=broad-except
                except Exception:
                    pass

        for switch in switches.values():
            thread: Thread = Thread(target=checkSwitch, args=(switch,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
