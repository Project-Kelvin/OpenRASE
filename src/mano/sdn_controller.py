"""
Defines the class that manipulates the Ryu SDN controller.
"""

from ipaddress import IPv4Address, IPv4Network
import re
from threading import Thread
from typing import Tuple
from requests import Response
import requests
from shared.models.embedding_graph import EmbeddingGraph, ForwardingLink
from shared.models.config import Config
from shared.utils.config import getConfig
from mininet.node import OVSKernelSwitch
from utils.ryu import getRyuRestUrl
from utils.tui import TUI

class SDNController():
    """
    Class that communicates with the Ryu SDN controller.
    """

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self._switchLinks: "dict[str, IPv4Address]" = {}

    def _installFlow(self, destination: IPv4Network, gateway: int, switch: OVSKernelSwitch, destinationMAC: str = None) -> None:
        """
        Install a flow in a switch.

        Parameters:
            destination (IPv4Network): The destination of the flow.
            gateway (int): The gateway of the flow.
            switch (OVSKernelSwitch): The switch to install the flow in.
            destinationMAC (str): The MAC address of the destination. `None` if the switch is not a host gateway.

        Raises:
            RuntimeError: If the flow could not be installed.
        """

        config: Config = getConfig()
        actions: "list[dict]" = [
            {
                "port": gateway,
                "type": "OUTPUT"
            }
        ]

        if destinationMAC is not None:
            actions.insert(0, {
                "type": "SET_FIELD",
                "field": "eth_dst",
                "value": destinationMAC
            })

        data = {
            "dpid": int(self._getSwitchID(switch.name)),
            "cookie": 1,
            "cookie_mask": 1,
            "hard_timeout": 3000,
            "priority": 1000,
            "match": {
                "ipv4_dst": str(destination),
                "eth_type": 2048
            },
            "instructions": [
                {
                    "type": "APPLY_ACTIONS",
                    "actions": actions
                }
            ]
        }

        try:
            response: Response = requests.request(
                method="POST",
                url=f"{getRyuRestUrl()}/flowentry/add",
                json=data,
                timeout=config["general"]["requestTimeout"]
            )

            if response.status_code != 200:
                TUI.appendToLog(
                    f"      Failed to install flow in switch {switch.name} to {destination} via port {gateway}.\n{str(response.content)}", True)
        except Exception as e:
            TUI.appendToLog(
                    f"      Failed to install flow in switch {switch.name} to {destination} via port {gateway}.\n{str(e)}", True)

        TUI.appendToLog(f"      Installed flow from {switch.name} to {destination} via port {gateway}.")

    def _deleteFlow(self, destination: IPv4Network, gateway: int, switch: OVSKernelSwitch, destinationMAC: str = None) -> None:
        """
        Delete a flow in a switch.

        Parameters:
            destination (IPv4Network): The destination of the flow.
            gateway (int): The gateway of the flow.
            switch (OVSKernelSwitch): The switch to install the flow in.
            destinationMAC (str): The MAC address of the destination. `None` if the switch is not a host gateway.

        Raises:
            RuntimeError: If the flow could not be installed.
        """

        config: Config = getConfig()
        actions: "list[dict]" = [
            {
                "port": gateway,
                "type": "OUTPUT"
            }
        ]

        if destinationMAC is not None:
            actions.insert(0, {
                "type": "SET_FIELD",
                "field": "eth_dst",
                "value": destinationMAC
            })

        data = {
            "dpid": int(self._getSwitchID(switch.name)),
            "cookie": 1,
            "cookie_mask": 1,
            "hard_timeout": 3000,
            "priority": 1000,
            "match": {
                "ipv4_dst": str(destination),
                "eth_type": 2048
            },
            "instructions": [
                {
                    "type": "APPLY_ACTIONS",
                    "actions": actions
                }
            ]
        }

        try:
            response: Response = requests.request(
                method="POST",
                url=f"{getRyuRestUrl()}/flowentry/delete",
                json=data,
                timeout=config["general"]["requestTimeout"]
            )
            if response.status_code != 200:
                TUI.appendToLog(
                    f"      Failed to delete flow in switch {switch.name} to {destination} via port {gateway}.\{str(response.content)}", True)
        except Exception as e:
            TUI.appendToLog(
                    f"      Failed to delete flow in switch {switch.name} to {destination} via port {gateway}.\n{str(e)}", True)

        TUI.appendToLog(f"      Deleted flow from {switch.name} to {destination} via port {gateway}.")

    def installFlows(self, eg: EmbeddingGraph,
                     vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]",
                     switches: "dict[str, OVSKernelSwitch]",
                     linkedPorts: "dict[str, tuple[int, int]]",
                     hostMACs: "dict[str, str]") -> EmbeddingGraph:
        """
        Install flows in the switches in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to install flows in.
            vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address]]):
            The hosts of the VNFs in the embedding graph.
            switches (dict[str, OVSKernelSwitch]"): The switches to install flows in.
            linkedPorts (dict[str, tuple[int, int]]): The ports that are linked between switches.
            hostMACs (dict[str, str]): The MAC addresses of the hosts in the topology.

        Returns:
            EmbeddingGraph: The embedding graph with the flows installed.
        """

        links: "list[ForwardingLink]" = eg["links"]

        for link in links:
            sourceNetwork: IPv4Network = vnfHosts[link["source"]["id"]][0]
            link["source"]["ip"] = str(vnfHosts[link["source"]["id"]][1])
            destinationNetwork: IPv4Network = vnfHosts[link["destination"]["id"]][0]
            link["destination"]["ip"] = str(vnfHosts[link["destination"]["id"]][1])

            for index, switch in enumerate(link["links"]):
                nextSwitch: str = link["links"][index + 1] if index < len(link["links"]) - 1 else None
                prevSwitch: str = link["links"][index - 1] if index > 0 else None

                if nextSwitch is None:
                    self._installFlow(destinationNetwork, linkedPorts[f"{switch}-{link['destination']['id']}"][0],
                                        switches[switch], hostMACs[link["destination"]["id"]])
                else:
                    self._installFlow(destinationNetwork, linkedPorts[f"{switch}-{nextSwitch}"][0],
                                        switches[switch])
                if prevSwitch is None:
                    self._installFlow(sourceNetwork, linkedPorts[f"{switch}-{link['source']['id']}"][0],
                                        switches[switch], hostMACs[link["source"]["id"]])
                else:
                    self._installFlow(sourceNetwork, linkedPorts[f"{switch}-{prevSwitch}"][0],
                                        switches[switch])

        return eg

    def deleteFlows(self, eg: EmbeddingGraph,
                     vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]",
                     switches: "dict[str, OVSKernelSwitch]",
                     linkedPorts: "dict[str, tuple[int, int]]",
                     hostMACs: "dict[str, str]") -> EmbeddingGraph:
        """
        Delete flows in the switches in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to install flows in.
            vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address]]):
            The hosts of the VNFs in the embedding graph.
            switches (dict[str, OVSKernelSwitch]"): The switches to install flows in.
            linkedPorts (dict[str, tuple[int, int]]): The ports that are linked between switches.
            hostMACs (dict[str, str]): The MAC addresses of the hosts in the topology.

        Returns:
            EmbeddingGraph: The embedding graph with the flows installed.
        """

        links: "list[ForwardingLink]" = eg["links"]

        for link in links:
            sourceNetwork: IPv4Network = vnfHosts[link["source"]["id"]][0]
            link["source"]["ip"] = str(vnfHosts[link["source"]["id"]][1])
            destinationNetwork: IPv4Network = vnfHosts[link["destination"]["id"]][0]
            link["destination"]["ip"] = str(vnfHosts[link["destination"]["id"]][1])

            for index, switch in enumerate(link["links"]):
                nextSwitch: str = link["links"][index + 1] if index < len(link["links"]) - 1 else None
                prevSwitch: str = link["links"][index - 1] if index > 0 else None

                if nextSwitch is None:
                    self._deleteFlow(destinationNetwork, linkedPorts[f"{switch}-{link['destination']['id']}"][0],
                                        switches[switch], hostMACs[link["destination"]["id"]])
                else:
                    self._deleteFlow(destinationNetwork, linkedPorts[f"{switch}-{nextSwitch}"][0],
                                        switches[switch])
                if prevSwitch is None:
                    self._deleteFlow(sourceNetwork, linkedPorts[f"{switch}-{link['source']['id']}"][0],
                                        switches[switch], hostMACs[link["source"]["id"]])
                else:
                    self._deleteFlow(sourceNetwork, linkedPorts[f"{switch}-{prevSwitch}"][0],
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
                        url=f"{getRyuRestUrl()}/flow/{self._getSwitchID(switch.name)}",
                        timeout=config["general"]["requestTimeout"]
                    )

                    if response.status_code == 200:
                        isReady = True
                        TUI.appendToLog(f"  {switch.name} is ready.")
                except:
                    pass

        for switch in switches.values():
            thread: Thread = Thread(target=checkSwitch, args=(switch,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def _getSwitchID(self, name: str) -> int:
        """
        Get the ID of a switch.

        Parameters:
            name (str): The name of the switch.

        Returns:
            int: The ID of the switch.
        """

        return int(re.findall(r'\d+', name)[0])
