"""
This defines the logic to check the liveness of the Mininet environment.
"""

from shared.models.topology import Host, Topology
from threading import Thread
from time import sleep
from typing import Any
from docker import DockerClient, from_env
from shared.utils.config import getConfig
from utils.host import addHostNode
from utils.tui import TUI


def isHostAlive(host: Host) -> bool:
    """
    Check if the host is alive.

    Parameters:
        host (Host): The host to check.

    Returns:
        bool: Whether the host is alive.
    """

    try:
        client: DockerClient = from_env()
        container = client.containers.get(f"mn.{host['id']}Node")

        if container.status == "running":
            return True

        return False
    except Exception:
        return False


def hostHeartBeat(host: Host, net: Any, stop: "list[bool]") -> None:
    """
    Send a heartbeat to the host.

    Parameters:
        host (Host): The host to send the heartbeat to.
        net (Any): The Mininet environment.
        stop (list[bool]): The stop flag.
    """

    while not stop[0]:
        if not isHostAlive(host):
            TUI.appendToLog(f"Host {host['id']} is down. Restarting host.")
            try:
                oldHost: Any = net.get(f"{host['id']}Node")
                sff: Any = net.get(f"{host['id']}")
                try:
                    net.delLinkBetween(sff, oldHost)
                except Exception:
                    TUI.appendToLog(f"Link between {host['id']} and {host['id']}Node doesn't exist.")
                try:
                    net.delHost(oldHost)
                except Exception:
                    TUI.appendToLog(f"Host {host['id']}Node cannot be deleted.")
                client: DockerClient = from_env()
                try:
                    container = client.containers.get(f"mn.{host['id']}Node")
                    container.remove(force=True)
                except Exception:
                    TUI.appendToLog(f"Container {host['id']}Node doesn't exist. Proceeding with restart.")
                hostNode: Any = addHostNode(host, net)
                link: Any = net.addLink(sff, hostNode)
                hostNode.cmd(
                    f"ip addr add {getConfig()['sff']['network1']['hostIP']}/{getConfig()['sff']['network1']['mask']} dev {hostNode.name}-eth0"
                )
                hostNode.cmd(
                    f"ip addr add {getConfig()['sff']['network2']['hostIP']}/{getConfig()['sff']['network2']['mask']} dev {hostNode.name}-eth0"
                )
                sff.cmd(
                    f"ip addr add {getConfig()['sff']['network1']['sffIP']}/{getConfig()['sff']['network1']['mask']} dev {link.intf1.name}"
                )
                sff.cmd(
                    f"ip addr add {getConfig()['sff']['network2']['sffIP']}/{getConfig()['sff']['network2']['mask']} dev {link.intf1.name}"
                )

            except Exception as e:
                TUI.appendToLog(f"Failed to restart host {host['id']}: {e}")

        sleep(2)


def checkLiveness(net: Any, topology: Topology, stop: "list[bool]") -> None:
    """
    Check the liveness of the Mininet environment.

    Parameters:
        net (Any): The Mininet environment.
        topology (Topology): The topology.
        stop (list[bool]): The stop flag.
    """

    for host in topology["hosts"]:
        thread = Thread(target=hostHeartBeat, args=(host, net, stop))
        thread.start()
