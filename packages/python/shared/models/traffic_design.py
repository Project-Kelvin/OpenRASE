"""
Defines the models associated with the traffic design.
"""

from typing import List


class TrafficStep(dict):
    """
    Defines the `TrafficDesign` dictionary type.
    """

    target: int
    duration: str


TrafficDesign = List[TrafficStep]
