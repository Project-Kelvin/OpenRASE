"""
This defines utils used for data manipulation.
"""

import pandas as pd
from models.telemetry import HostData, SingleHostData
import docker


def hostDataToFrame(hostData: "list[HostData]") -> "pd.DataFrame":
    """
    Converts the host data to a pandas DataFrame.
    """

    columns: "list[str]" = ["startTime", "endTime", "cpu", "memory"]
    data: "list[list[str]]" = []

    for host in hostData:
        firstHostData: SingleHostData = list(host["hostData"].values())[0]
        row: "list[str]" = [
            host["startTime"],
            host["endTime"],
            firstHostData["cpuUsage"][0],
            firstHostData["memoryUsage"][0],
        ]
        data.append(row)

    return pd.DataFrame(data, columns=columns)


def mergeHostAndTrafficData(
    hostData: "pd.DataFrame", trafficData: "pd.DataFrame"
) -> "pd.DataFrame":
    """
    Merges the host and traffic data.
    """

    medians: "list[float]" = []
    reqpss: "list[float]" = []
    rowsToDelete: "list[int]" = []
    for index, row in hostData.iterrows():
        startTime: int = int(row["startTime"]) * 1000000000  # to match Flux output
        endTime: int = int(row["endTime"]) * 1000000000
        duration: int = int((endTime - startTime) / 1000000000)

        rangeData: pd.DataFrame = trafficData[
            (trafficData["_time"] >= startTime) & (trafficData["_time"] < endTime)
        ]

        if rangeData.empty:
            rowsToDelete.append(index)
        else:
            median: float = rangeData["_value"].median()
            reqps: float = float(len(rangeData) / duration)

            medians.append(median)
            reqpss.append(reqps)

    hostData = hostData.drop(rowsToDelete)
    hostData["median"] = medians
    hostData["reqps"] = reqpss
    hostData["duration"] = hostData["endTime"] - hostData["startTime"]

    return hostData


def getAvailableCPUAndMemory() -> tuple[float, float]:
    """
    Returns the available CPU and memory in the host server.

    Returns:
        tuple[float, float]: Maximum CPU and memory usage.
    """

    client = docker.from_env()
    info = client.info()
    maxCPU = float(info.get("NCPU", 0))
    maxMemory = float(info.get("MemTotal", 0)) / (1024 * 1024)  # Convert bytes to MB

    return maxCPU, maxMemory
