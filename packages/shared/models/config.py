"""
Defines the models of the `Config` dictionary type and its associated types.
"""


class SFFNetwork(dict):
    """
    Defines the `SFFNetwork` dictionary type.
    """

    networkIP: str
    sffIP: str
    hostIP: str
    mask: int


class SFF(dict):
    """
    Defines the `SFF` dictionary type.
    """

    network1: SFFNetwork
    network2: SFFNetwork
    port: int


class Server(dict):
    """
    Defines the `WebServer` dictionary type.
    """

    port: int


class VNFProxy(dict):
    """
    Defines the `VNFProxy` dictionary type.
    """

    port: int


class SFCClassifier(dict):
    """
    Defines the `SFCClassifier` dictionary type.
    """

    port: int


class General(dict):
    """
    Defines the `General` dictionary type.
    """

    requestTimeout: int


class IPRange(dict):
    """
    Defines the `IPRange` dictionary type.
    """

    mask: int


class VNFs(dict):
    """
    Defines the `VNFs` dictionary type.
    """

    sharedVolumes: "dict[str, list[str]]"


class Config(dict):
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
    ipRange: IPRange
    vnfs: VNFs
