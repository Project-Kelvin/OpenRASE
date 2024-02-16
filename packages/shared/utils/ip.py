"""
Provides utils replated IP address operations.
"""


from typing import Iterator, Literal, Tuple
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network

from shared.models.config import Config
from shared.utils.config import getConfig

def checkIPBelongsToNetwork(ip: str, networkIP: str) -> bool:
    """
    Check whether the given IP address belongs to the given network.

    Parameters:
        ip (str): The IP address to be checked.
        networkIP (str): The network to be checked.

    Returns:
        bool: True if the IP address belongs to the network, False otherwise.
    """

    return ip_address(ip) in ip_network(networkIP)

def incrementNetworkIP(ip: IPv4Network) -> IPv4Network:
    """
    Increment the given IP address.

    Parameters:
        ip (IPv4Network): The IP address to be incremented.

    Returns:
        IPv4Network: The incremented IP address.
    """

    ipInt: int = int(ip.network_address)
    incrementor: int = 2 ** (32 - ip.prefixlen)

    return ip_network((ipInt + incrementor, ip.prefixlen))

def generateLocalNetworkIP(mask: int, existingIPs: "list[IPv4Network]") -> IPv4Network:
    """
    Generate a local network IP address.

    Parameters:
        mask (int): The mask of the network.
        existingIPs (list[IPv4Network]): The list of existing IPs in the network.

    Returns:
        IPv4Network: The generated network IP address.

    Raises:
        ValueError: If the IP class is invalid.
        RuntimeError: If no more IPs are available in the network.
    """

    if mask < 8 or mask > 30:
        raise ValueError(f"Invalid mask: {mask}")

    classes: "list[str]" = ["A", "B", "C"]

    # pylint: disable=invalid-name
    ClassRange = dict(
        "ClassRange", {"start": IPv4Network, "end": IPv4Network})

    classRanges: "dict[Literal['A', 'B', 'C'], ClassRange]" = {
        "A": {
            "start": ip_network(("10.0.0.0", mask)),
            "end": ip_network(("10.255.255.255", mask), strict=False)
        },
        "B": {
            "start": ip_network(("172.16.0.0", mask)),
            "end": ip_network(("172.31.255.255", mask), strict=False)
        },
        "C": {
            "start": ip_network(("192.168.0.0", mask)),
            "end": ip_network(("192.168.255.255", mask), strict=False)
        }
    }

    ipAddr: IPv4Network = None
    currentIPClass: str = ""
    nextIP: IPv4Network = None

    if len(existingIPs) == 0:
        ipAddr = classRanges["A"]["start"]
        currentIPClass = "A"
        nextIP = ipAddr
    else:
        ipAddr = existingIPs[-1]
        nextIP = incrementNetworkIP(ipAddr)

        if (ipAddr >= classRanges["A"]["start"] and ipAddr <= classRanges["A"]["end"]):
            currentIPClass = "A"
        elif (ipAddr >= classRanges["B"]["start"] and ipAddr <= classRanges["B"]["end"]):
            currentIPClass = "B"
        elif (ipAddr >= classRanges["C"]["start"] and ipAddr <= classRanges["C"]["end"]):
            currentIPClass = "C"

    if nextIP > classRanges[currentIPClass]["end"]:
        if classes.index(currentIPClass) == len(classes) - 1:
            raise RuntimeError("No more IPs available in the network.")
        else:
            nextIP = ip_network(
                (classRanges[classes[classes.index(currentIPClass) + 1]]["start"].network_address, mask))

    return nextIP


def generateIP(existingIPs: "list[IPv4Network]") -> "Tuple[IPv4Network, IPv4Address, IPv4Address]":
    """
    Generate an IP address for the network.

    Parameters:
        existingIPs (list[IPv4Network]): The list of existing IPs in the network.

    Returns:
        Tuple[IPv4Network, IPv4Address, IPv4Address]: The generated IP address and the first two host IPs.
    """

    config: Config = getConfig()
    mask: int = config['ipRange']['mask']

    ip: IPv4Network = generateLocalNetworkIP(mask,existingIPs)
    existingIPs.append(ip)
    hosts: "Iterator[IPv4Address]" = ip.hosts()
    ip1: IPv4Address = next(hosts)
    ip2: IPv4Address = next(hosts)

    return (ip, ip1, ip2)
