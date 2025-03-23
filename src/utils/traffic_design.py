"""
This defines utils associated with traffic design.
"""

import numpy as np
from shared.models.traffic_design import TrafficDesign


def generateTrafficDesign(dataFile: str, scale: float = 1, hourDuration: float = 4, minimal: bool = False) -> "TrafficDesign":
    """
    Generate the Traffic Design.

    Parameters:
        dataFile (str): the CSV data file.
        scale (float): the scale factor.
        hourDuration (float): the simulated duration of the hour in seconds.
        minimal (bool): whether to generate minimal traffic design.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    design: TrafficDesign = []

    with open(dataFile, "r", encoding="utf8") as file:
        _header = file.readline()  # Read and discard the header
        lastReqps: int = 0
        for line in file:
            numOfReqs: int = int(line)
            # scale
            numOfReqs = round(numOfReqs * scale)
            if minimal and lastReqps == numOfReqs:
                continue
            lastReqps = numOfReqs
            #Considering an hour is equal to `hourDuration` seconds
            rate: int = numOfReqs // hourDuration

            for _ in range(hourDuration):
                design.append({
                    "target": rate,
                    "duration": "1s"
                })

    return design

def calculateTrafficDuration(trafficDesign: TrafficDesign) -> int:
    """
    Calculate the duration of the traffic.

    Parameters:
        trafficDesign (TrafficDesign): The traffic design.

    Returns:
        int: The duration of the traffic.
    """

    totalDuration: int = 0

    for traffic in trafficDesign:
        durationText: str = traffic["duration"]
        unit: str  = durationText[-1]
        if unit == "s":
            totalDuration += int(durationText[:-1])
        elif unit == "m":
            totalDuration += int(durationText[:-1]) * 60
        elif unit == "h":
            totalDuration += int(durationText[:-1]) * 60 * 60

    return totalDuration

def getTrafficDesignRate(trafficDesign: TrafficDesign, durations: "list[int]") -> "list[float]":
    """
    Get the traffic design rate.

    Parameters:
        trafficDesign (TrafficDesign): The traffic design.
        durations (list[int]): The durations.

    Returns:
        list[float]: The traffic design rate.
    """

    rate: "list[float]" = [traffic["target"] for traffic in trafficDesign]
    currRate: int = 1

    reqps: "list[float]" = []

    for duration in durations:
        reqps.append(np.mean(rate[:duration]) if len(rate) != 0 else currRate)
        rate = rate[duration:]

    return reqps
