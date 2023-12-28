"""
Defines the models of the `Config` dictionary type and its associated types.
"""

from typing import TypedDict


class SFFNetwork(TypedDict):
    """
    Defines the `SFFNetwork` dictionary type.
    """

    networkIP: str
    sffIP: str


class SFF(TypedDict):
    """
    Defines the `SFF` dictionary type.
    """

    network1: SFFNetwork
    network2IP: SFFNetwork
    port: int


class Server(TypedDict):
    """
    Defines the `WebServer` dictionary type.
    """

    port: int


class VNFProxy(TypedDict):
    """
    Defines the `VNFProxy` dictionary type.
    """

    port: int


class SFCClassifier(TypedDict):
    """
    Defines the `SFCClassifier` dictionary type.
    """

    port: int

class General(TypedDict):
    """
    Defines the `General` dictionary type.
    """

    port: int

class Config(TypedDict):
    """
    Defines the `Config` dictionary type.
    """

    sff: SFF
    server: Server
    vnfProxy: VNFProxy
    sfcClassifier: SFCClassifier
    general: General
    repoAbsolutePath: str
    templates: "list[str]"
