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


class SDNController:
    """
    Class that communicates with the Ryu SDN controller.
    """

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self._switchLinks: "dict[str, IPv4Address]" = {}

    def _configureSwitchFlow(
        self,
        installFlow: bool,
        destination: IPv4Network,
        gateway: int,
        switch: OVSKernelSwitch,
        destinationMAC: str = None,
        destinationIP: IPv4Address = None,
        sourceIP: str = None,
        sourceMAC: str = None,
    ) -> None:
        """
        Install a flow in a switch.

        Parameters:
            installFlow (bool): Whether to install the flow or delete it.
            destination (IPv4Network): The destination of the flow.
            gateway (int): The gateway of the flow.
            switch (OVSKernelSwitch): The switch to install the flow in.
            destinationMAC (str): The MAC address of the destination. `None` if the switch is not a host gateway.
            destinationIP (IPv4Address): The first host IP address of the destination.
            sourceIP (str): The source IP address of the flow. `None` if the switch is not a host gateway.
            sourceMAC (str): The MAC address of the source. `None` if the switch is not a host gateway.

        Raises:
            RuntimeError: If the flow could not be installed.
        """

        config: Config = getConfig()
        actions: "list[dict]" = []

        if destinationMAC is not None:
            actions.append(
                {"type": "SET_FIELD", "field": "eth_dst", "value": destinationMAC}
            )

        if destinationIP is not None:
            actions.append(
                {
                    "type": "SET_FIELD",
                    "field": "ipv4_dst",
                    "value": f"{str(destinationIP)}",
                }
            )

        if sourceIP is not None and sourceMAC is not None:
            actions.append(
                {"type": "SET_FIELD", "field": "ipv4_src", "value": sourceIP}
            )

        actions.append({"port": gateway, "type": "OUTPUT"})

        data = {
            "dpid": int(self._getSwitchID(switch.name)),
            "cookie": 1,
            "cookie_mask": 1,
            "hard_timeout": 3000,
            "priority": 1000,
            "match": {"ipv4_dst": str(destination), "eth_type": 2048},
            "instructions": [{"type": "APPLY_ACTIONS", "actions": actions}],
        }

        if sourceMAC is not None:
            data["match"]["eth_src"] = sourceMAC
        elif sourceIP is not None:
            data["match"]["ipv4_src"] = sourceIP

        operation: str = "install" if installFlow else "delete"
        operationPast: str = "Installed" if installFlow else "Deleted"

        try:
            response: Response = requests.request(
                method="POST",
                url=f"{getRyuRestUrl()}{'/flowentry/add' if installFlow else '/flowentry/delete'}",
                json=data,
                timeout=config["general"]["requestTimeout"],
            )

            if response.status_code != 200:
                TUI.appendToLog(
                    f"      Failed to {operation} flow in switch {switch.name} to {destination} via port {gateway}.\n{str(response.content)}",
                    True,
                )
        except Exception as e:
            TUI.appendToLog(
                f"      Failed to {operation} flow in switch {switch.name} to {destination} via port {gateway}.\n{str(e)}",
                True,
            )

        TUI.appendToLog(
            f"      {operationPast} flow from {switch.name} to {destination} via port {gateway}."
        )

    def configureSwitchFlows(
        self,
        eg: EmbeddingGraph,
        vnfHosts: "dict[str, Tuple[IPv4Network, IPv4Address]]",
        switches: "dict[str, OVSKernelSwitch]",
        linkedPorts: "dict[str, tuple[int, int]]",
        hostMACs: "dict[str, str]",
        firstHostIPs: "dict[str, tuple[IPv4Network, IPv4Address]]",
        installFlow: bool = True,
    ) -> EmbeddingGraph:
        """
        Install flows in the switches in the topology.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to install flows in.
            vnfHosts (dict[str, Tuple[IPv4Network, IPv4Address]]):
            The hosts of the VNFs in the embedding graph.
            switches (dict[str, OVSKernelSwitch]"): The switches to install flows in.
            linkedPorts (dict[str, tuple[int, int]]): The ports that are linked between switches.
            hostMACs (dict[str, str]): The MAC addresses of the hosts in the topology.
            firstHostIPs (dict[str, tuple[IPv4Network, IPv4Address]]): The first IP addresses of the hosts in the topology.
            installFlow (bool): Whether to install the flow or delete it.

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
                nextSwitch: str = (
                    link["links"][index + 1] if index < len(link["links"]) - 1 else None
                )
                prevSwitch: str = link["links"][index - 1] if index > 0 else None

                if nextSwitch is None and prevSwitch is None:
                    self._configureSwitchFlow(
                        installFlow,
                        destinationNetwork,
                        linkedPorts[f"{switch}-{link['destination']['id']}"][0],
                        switches[switch],
                        hostMACs[link["destination"]["id"]],
                        firstHostIPs[link["destination"]["id"]][1],
                        link["source"]["ip"],
                        hostMACs[link["source"]["id"]],
                    )
                    self._configureSwitchFlow(
                        installFlow,
                        sourceNetwork,
                        linkedPorts[f"{switch}-{link['source']['id']}"][0],
                        switches[switch],
                        hostMACs[link["source"]["id"]],
                        firstHostIPs[link["source"]["id"]][1],
                        link["destination"]["ip"],
                        hostMACs[link["destination"]["id"]],
                    )
                elif nextSwitch is None:
                    self._configureSwitchFlow(
                        installFlow,
                        destinationNetwork,
                        linkedPorts[f"{switch}-{link['destination']['id']}"][0],
                        switches[switch],
                        hostMACs[link["destination"]["id"]],
                        firstHostIPs[link["destination"]["id"]][1],
                        link["source"]["ip"],
                    )
                    self._configureSwitchFlow(
                        installFlow,
                        sourceNetwork,
                        linkedPorts[f"{switch}-{prevSwitch}"][0],
                        switches[switch],
                        None,
                        None,
                        link["destination"]["ip"],
                        hostMACs[link["destination"]["id"]],
                    )
                elif prevSwitch is None:
                    self._configureSwitchFlow(
                        installFlow,
                        sourceNetwork,
                        linkedPorts[f"{switch}-{link['source']['id']}"][0],
                        switches[switch],
                        hostMACs[link["source"]["id"]],
                        firstHostIPs[link["source"]["id"]][1],
                        link["destination"]["ip"],
                    )
                    self._configureSwitchFlow(
                        installFlow,
                        destinationNetwork,
                        linkedPorts[f"{switch}-{nextSwitch}"][0],
                        switches[switch],
                        None,
                        None,
                        link["source"]["ip"],
                        hostMACs[link["source"]["id"]],
                    )
                else:
                    self._configureSwitchFlow(
                        installFlow,
                        destinationNetwork,
                        linkedPorts[f"{switch}-{nextSwitch}"][0],
                        switches[switch],
                        None,
                        None,
                        link["source"]["ip"],
                    )
                    self._configureSwitchFlow(
                        installFlow,
                        sourceNetwork,
                        linkedPorts[f"{switch}-{prevSwitch}"][0],
                        switches[switch],
                        None,
                        None,
                        link["destination"]["ip"],
                    )

        return eg

    def waitTillReady(self, switches: "dict[str, OVSKernelSwitch]") -> None:
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
                        timeout=config["general"]["requestTimeout"],
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

        return int(re.findall(r"\d+", name)[0])
