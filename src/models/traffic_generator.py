"""
Contains the models associated with traffic generation.
"""

class TrafficData(dict):
    """
    Represents the data of teh traffic.
    """

    httpReqs = int
    averageLatency = float

