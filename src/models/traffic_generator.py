"""
Contains the models associated with traffic generation.
"""

class TrafficData(dict):
    """
    Represents the data of the traffic.
    """
    metric: str
    value: float
    timestamp: int
