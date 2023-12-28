"""
Provides utils replated IP address operations.
"""


def getMaskFromIP(ip: str) -> int:
    """
    Get the network mask from the given IP address.

    Parameters:
        ip (str): The IP address to get the network mask from.

    Returns:
        int: The network mask.
    """

    mask: str = ip.split("/")[1]

    return int(mask)


def getNetworkPartFromIP(ip: str) -> str:
    """
    Get the network part of the IP from the given IP address.
    Assumes that the mask is a multiple of 8.

    Parameters:
        ip (str): The IP address to get the network IP from.

    Returns:
        str: The network part of the IP.
    """

    networkIP: str = ip.split("/")[0]
    mask: int = getMaskFromIP(ip)
    ipBlocks: "list[str]" = networkIP.split(".")
    networkPart: str = ".".join(ipBlocks[0:mask // 8])

    return networkPart


def checkIPBelongsToNetwork(ip: str, networkIP: str) -> bool:
    """
    Check whether the given IP address belongs to the given network.

    Parameters:
        ip (str): The IP address to be checked.
        networkIP (str): The network to be checked.

    Returns:
        bool: True if the IP address belongs to the network, False otherwise.
    """

    networkPart: str = getNetworkPartFromIP(networkIP)

    return ip.startswith(networkPart)
