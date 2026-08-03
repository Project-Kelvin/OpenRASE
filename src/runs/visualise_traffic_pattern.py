""""
Turns the traffic patterns used for experiments into a line plot.
"""

import os
import matplotlib.pyplot as plt
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from utils.traffic_design import generateTrafficDesignFromFile

artifacts_dir = os.path.join(getConfig()['repoAbsolutePath'], "artifacts", "traffic")
if not os.path.isdir(artifacts_dir):
    os.makedirs(artifacts_dir)

trafficDesign1: "list[TrafficDesign]" = [
    generateTrafficDesignFromFile(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "ga_dijkstra_algorithm",
            "data",
            "requests.csv",
        ),
        0.1,
        1,
        False,
        False,
    )
]

trafficDesign2: "list[TrafficDesign]" = [
    generateTrafficDesignFromFile(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "ga_dijkstra_algorithm",
            "data",
            "requests.csv",
        ),
        0.1,
        1,
        False,
        True,
    )
]

trafficDesign3: "list[TrafficDesign]" = [
    generateTrafficDesignFromFile(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "ga_dijkstra_algorithm",
            "data",
            "requests.csv",
        ),
        0.2,
        1,
        False,
        False,
    )
]

def run() -> None:
    """
    Runs the traffic pattern visualisation.
    """

    reqps1: list[float] = [req["target"] for req in trafficDesign1[0]]
    reqps2: list[float] = [req["target"] for req in trafficDesign2[0]]
    reqps3: list[float] = [req["target"] for req in trafficDesign3[0]]

    plt.figure(2, (14, 6), dpi=300)
    plt.plot(reqps1, label="Traffic Pattern A scaled at 1", color="red")
    plt.plot(reqps2, label="Traffic Pattern B scaled at 1", color="blue")
    plt.plot(reqps3, label="Traffic Pattern A scaled at 2", color="green")
    plt.xlabel("Time(h)")
    plt.ylabel("Requests per Second")
    plt.title("Traffic Patterns Used for Evaluation")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(artifacts_dir, "traffic_pattern.png"))
    plt.clf()
