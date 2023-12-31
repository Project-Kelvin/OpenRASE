"""
Provides utils replated IP address operations.
"""


from typing import Literal, TypedDict
from ipaddress import IPv4Network, ip_address, ip_network

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
    ClassRange = TypedDict(
        "ClassRange", {"start": IPv4Network, "end": IPv4Network})

    classAEnd = int(ip_address("10.255.255.255")) - \
        2 ** (32 - mask) + 1
    classBEnd = int(ip_address("172.31.255.255")) - \
        2 ** (32 - mask) + 1
    classCEnd = int(ip_address(
        "192.168.255.255")) - 2 ** (32 - mask) + 1

    classRanges: "dict[Literal['A', 'B', 'C'], ClassRange]" = {
        "A": {
            "start": ip_network(("10.0.0.0", mask)),
            "end": ip_network((classAEnd, mask))
        },
        "B": {
            "start": ip_network(("172.16.0.0", mask)),
            "end": ip_network((classBEnd, mask))
        },
        "C": {
            "start": ip_network(("192.168.0.0", mask)),
            "end": ip_network((classCEnd, mask))
        }
    }

    ipAddr: IPv4Network = None
    currentIPClass: str = ""
    nextIP: IPv4Network = None

    if len(existingIPs) == 0:
        ipAddr = classRanges["A"]["start"]
        currentIPClass = "A"
        ipInt: int = int(ipAddr.network_address)
        nextIP: IPv4Network = ip_network((ipInt, mask))
    else:
        ipAddr = existingIPs[-1]
        ipInt: int = int(ipAddr.network_address)
        incrementor: int = 2 ** (32 - mask)
        nextIP: IPv4Network = ip_network(
            (ipInt + incrementor, mask))

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
