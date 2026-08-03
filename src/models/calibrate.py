"""
This defines the models associated with calibration.
"""

class ResourceDemand(dict):
    """
    Defines the resource demand of a VNF.
    """

    cpu: float
    memory: float
    power: float
