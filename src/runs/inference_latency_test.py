"""
This defines the test to evaluate the inference latency of the VNF CPU/memory usage predictors.
"""

import os
import random
from timeit import default_timer
from typing import Any

import numpy as np
import pandas as pd
from shared.utils.config import getConfig

from algorithms.hybrid.surrogate.surrogate import predictLatency
from calibrate.demand_predictor import DemandPredictor

artifactsDir: str = os.path.join(getConfig()["repoAbsolutePath"], "artifacts")

if not os.path.exists(artifactsDir):
    os.makedirs(artifactsDir)

experimentsDir: str = os.path.join(artifactsDir, "experiments")

if not os.path.exists(experimentsDir):
    os.makedirs(experimentsDir)

inferenceLatencyDir: str = os.path.join(experimentsDir, "inference_latency")

if not os.path.exists(inferenceLatencyDir):
    os.makedirs(inferenceLatencyDir)

demandPredictor: DemandPredictor = DemandPredictor()
vnfModels: dict[str, dict[str, Any]] = demandPredictor.getVNFModels()


def inferVNFTime(data: list[float], vnf: str, metric: str) -> float:
    """
    Get the inference time for the VNF.

    Parameters:
        data (list[float]): The data to infer the resource demand for.
        vnf (str): The VNF to infer the resource demand for.
        metric (str): The metric to infer the resource demand for.

    Returns:
        float: The inference time for the VNF.
    """

    model: Any = vnfModels[vnf][metric]
    startTime: float = default_timer()
    prediction: list[float] = model.predict(np.array(data), verbose=0)[0]
    endTime: float = default_timer()

    return round((endTime - startTime) * 1000, 2)

def inferApproximatorTime(data: pd.DataFrame) -> float:
    """
    Get the inference time for the approximator.

    Parameters:
        data (pd.DataFrame): The data to infer the resource demand for.

    Returns:
        float: The inference time for the approximator.
    """

    startTime: float = default_timer()
    predictLatency(data)
    endTime: float = default_timer()

    return round((endTime - startTime) * 1000, 2)

def run() -> None:
    """
    Run the inference latency test.

    Returns:
        None
    """

    data: list[list[float]] = []
    noOfRows: int = 100
    noOfExperiments: int = 50
    vnf: list[str] = getConfig()["vnfs"]["names"]
    metrics: list[str] = ["cpu", "memory"]
    latencies: dict[str, dict[str, list[float]]] = {
        vnfName: {metric: [] for metric in metrics} for vnfName in vnf}

    for _i in range(noOfExperiments):
        row: list[float] = [random.uniform(1, 500) for _i in range(noOfRows)]
        data.append(row)

    for vnfName in vnf:
        for metric in metrics:
            latencies[vnfName][metric] = []
            for exp in data:
                latencies[vnfName][metric].append(
                    inferVNFTime(exp, vnfName, metric))

    print("Inference Latencies (in milliseconds):")

    with open(os.path.join(inferenceLatencyDir, "inference_latencies.csv"), "w") as f:
        f.write("VNF,Metric,Mean,Median,Q1,Q3,Std,Min,Max\n")
        for vnfName in vnf:
            for metric in metrics:
                print(f"{vnfName} - {metric}: {latencies[vnfName][metric]}")
                f.write(
                    f"{vnfName},{metric},"
                    f"{np.mean(latencies[vnfName][metric]):.2f},"
                    f"{np.median(latencies[vnfName][metric]):.2f},"
                    f"{np.percentile(latencies[vnfName][metric], 25):.2f},"
                    f"{np.percentile(latencies[vnfName][metric], 75):.2f},"
                    f"{np.std(latencies[vnfName][metric]):.2f},"
                    f"{np.min(latencies[vnfName][metric]):.2f},"
                    f"{np.max(latencies[vnfName][metric]):.2f}\n"
                )


    approximatorData: pd.DataFrame = pd.DataFrame(
        {
            "max_cpu": [random.uniform(0, 1.75) for _ in range(noOfRows)],
            "max_link_score": [random.uniform(0, 300) for _ in range(noOfRows)],
        }
    )

    approximatorLatencies: list[float] = []

    for _i in range(noOfExperiments):
        approximatorLatencies.append(inferApproximatorTime(approximatorData))

    print(f"Approximator Inference Latencies (in milliseconds): {approximatorLatencies}")

    with open(os.path.join(inferenceLatencyDir, "approximator_inference_latencies.csv"), "w") as f:
        f.write("Mean,Median,Q1,Q3,Std,Min,Max\n")
        f.write(
            f"{np.mean(approximatorLatencies)},"
            f"{np.median(approximatorLatencies)},"
            f"{np.percentile(approximatorLatencies, 25)},"
            f"{np.percentile(approximatorLatencies, 75)},"
            f"{np.std(approximatorLatencies)},"
            f"{np.min(approximatorLatencies)},"
            f"{np.max(approximatorLatencies)}\n"
        )
