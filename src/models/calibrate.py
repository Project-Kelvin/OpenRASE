"""
This defines teh models associated with calibration.
"""

class ResourceDemand(dict):
    """
    Defines the resource demand of a VNF.
    """

    cpu: float
    memory: float
    ior: float
