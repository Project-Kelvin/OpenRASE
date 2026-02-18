"""
This defines utils associated with traffic design.
"""

import numpy as np
import polars as pl
from shared.models.traffic_design import TrafficDesign, TrafficStep

def generateTrafficDesign(start: int, end: int, duration: int) -> "TrafficDesign":
    """
    Generate the traffic rate.

    Parameters:
        start (int): the start rate.
        end (int): the end rate.
        duration (int): the duration.

    Returns:
        TrafficDesign: the traffic design with the rate.
    """

    rate: "TrafficDesign" = []

    for i in range(duration):
        req: int = start + round((end - start) * i / (duration - 1))
        rate.append({
            "target": req,
            "duration": "1s"
        })

    return rate

def generateTrafficDesignFromFile(dataFile: str, scale: int = 1, hourDuration: int = 4, minimal: bool = False, type2: bool = False) -> "TrafficDesign":
    """
    Generate the Traffic Design.

    Parameters:
        dataFile (str): the CSV data file.
        scale (int): the scale factor.
        hourDuration (int): the simulated duration of the hour in seconds.
        minimal (bool): whether to generate minimal traffic design.
        type2 (bool): whether to generate type 2 traffic design.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    design: TrafficDesign = []

    with open(dataFile, "r", encoding="utf8") as file:
        _header = file.readline() # Read and discard the header
        lastReqps: int = 0
        for line in file:
            numOfReqs: int = int(line)
            # scale
            numOfReqs = round(numOfReqs * scale)
            if minimal and lastReqps == numOfReqs:
                continue
            lastReqps = numOfReqs
            # Considering an hour is equal to `hourDuration` seconds
            rate: int = numOfReqs // hourDuration

            for _ in range(hourDuration):
                design.append({
                    "target": rate,
                    "duration": "1s"
                })

    if type2:
        midPoint: int = len(design) // 2
        firstHalf: TrafficDesign = design[:midPoint]
        secondHalf: TrafficDesign = design[midPoint:]
        newDesign: TrafficDesign = secondHalf + firstHalf

        return newDesign

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

def generateTrafficDesignFromIoTTrace(dataFile: str, hourDuration: int = 5*60, scale: int = 1000) -> TrafficDesign:
    """
    Generate the Traffic Design from an IoT trace.

    Parameters:
        dataFile (str): the CSV data file.
        hourDuration (int): the simulated duration of the hour in seconds.
        scale (int): the scale factor.

    Returns:
        TrafficDesign: the Traffic Design.
    """

    df: pl.DataFrame = (
        pl.read_csv(
            dataFile,
            has_header=False,
            new_columns=["timestamp"],
        )
        .with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime"))
        .group_by_dynamic("datetime", every="1h")
        .agg((pl.len() / scale).round(0).cast(pl.Int32).alias("count"))
        .sort("datetime")
    )

    counts: list[int] = df["count"].to_list()
    trafficDesign: TrafficDesign = []

    for count in counts:
        for _i in range(hourDuration):
            trafficDesign.append(TrafficStep(target=round(count), duration="1s"))

    return trafficDesign
