"""
Contains the models associated with traffic generation.
"""

class TrafficData(dict):
    """
    Represents the data of teh traffic.
    """

    httpReqs = int
    averageLatency = float
    variance= float
    q2 = float
    q3 = float
    q1 = float
    max = float
    min = float
