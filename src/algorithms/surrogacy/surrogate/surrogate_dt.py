"""
This defines the surrogate model based on Decision Trees.
"""

import os
import random
from typing import Any
import numpy as np
from shared.utils.config import getConfig
import ydf
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)


colsInclude: "list[str]" = ["link", "max_cpu"]
OUTPUT: str = "latency"
modelPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/surrogate"
dataPath: str = (
    getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/data/latency.csv"
)

highCpuHighLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_hc_hl.csv",
)
highCpuLowLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_hc_ll.csv",
)
lowCpuHighLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_lc_hl.csv",
)
lowCpuLowLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_lc_ll.csv",
)
highCpuLowLink2Path: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_hc_ll_2.csv",
)
highCpuHighLink2Path: str = os.path.join(
    getConfig()["repoAbsolutePath"],
    "src",
    "algorithms",
    "surrogacy",
    "data",
    "latency_hc_hl_2.csv",
)


def train() -> None:
    """
    Trains the model.
    """

    artifactsPath: str = (
        getConfig()["repoAbsolutePath"] + "/artifacts/experiments/surrogacy"
    )

    highCpuHighLinkData: pd.DataFrame = pd.read_csv(highCpuHighLinkPath, sep=r"\s*,\s*")
    highCpuLowLinkData: pd.DataFrame = pd.read_csv(highCpuLowLinkPath, sep=r"\s*,\s*")
    lowCpuHighLinkData: pd.DataFrame = pd.read_csv(lowCpuHighLinkPath, sep=r"\s*,\s*")
    lowCpuLowLinkData: pd.DataFrame = pd.read_csv(lowCpuLowLinkPath, sep=r"\s*,\s*")
    highCpuLowLink2Data: pd.DataFrame = pd.read_csv(highCpuLowLink2Path, sep=r"\s*,\s*")
    highCpuHighLink2Data: pd.DataFrame = pd.read_csv(
        highCpuHighLink2Path, sep=r"\s*,\s*"
    )

    highCpuHighLinkData.loc[highCpuHighLinkData[OUTPUT] == 0, OUTPUT] = 1500
    highCpuLowLinkData.loc[highCpuLowLinkData[OUTPUT] == 0, OUTPUT] = 1500
    lowCpuHighLinkData.loc[lowCpuHighLinkData[OUTPUT] == 0, OUTPUT] = 1500
    lowCpuLowLinkData.loc[lowCpuLowLinkData[OUTPUT] == 0, OUTPUT] = 1500
    highCpuLowLink2Data.loc[highCpuLowLink2Data[OUTPUT] == 0, OUTPUT] = 1500
    highCpuHighLink2Data.loc[highCpuHighLink2Data[OUTPUT] == 0, OUTPUT] = 1500

    highCpuHighLinkData = highCpuHighLinkData.groupby(["generation", "individual"]).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )
    highCpuLowLinkData = highCpuLowLinkData.groupby(["generation", "individual"]).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )
    lowCpuHighLinkData = lowCpuHighLinkData.groupby(["generation", "individual"]).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )
    lowCpuLowLinkData = lowCpuLowLinkData.groupby(["generation", "individual"]).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )
    highCpuLowLink2Data = highCpuLowLink2Data.groupby(["generation", "individual"]).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )
    highCpuHighLink2Data = highCpuHighLink2Data.groupby(
        ["generation", "individual"]
    ).agg(
        latency=("latency", "mean"), max_cpu=("max_cpu", "mean"), link=("link", "mean")
    )

    def clean(data: pd.DataFrame) -> pd.DataFrame:
        q1 = data[OUTPUT].quantile(0.25)
        q3 = data[OUTPUT].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        return data[(data[OUTPUT] > lower_bound) & (data[OUTPUT] < upper_bound)]

    highCpuHighLinkData = clean(highCpuHighLinkData)
    highCpuLowLinkData = clean(highCpuLowLinkData)
    lowCpuHighLinkData = clean(lowCpuHighLinkData)
    lowCpuLowLinkData = clean(lowCpuLowLinkData)
    highCpuLowLink2Data = clean(highCpuLowLink2Data)
    highCpuHighLink2Data = clean(highCpuHighLink2Data)

    highCpuHighLinkTrainData: pd.DataFrame = highCpuHighLinkData.sample(
        frac=0.8, random_state=0
    )
    highCpuLowLinkTrainData: pd.DataFrame = highCpuLowLinkData.sample(
        frac=0.8, random_state=0
    )
    lowCpuHighLinkTrainData: pd.DataFrame = lowCpuHighLinkData.sample(
        frac=0.8, random_state=0
    )
    lowCpuLowLinkTrainData: pd.DataFrame = lowCpuLowLinkData.sample(
        frac=0.8, random_state=0
    )
    highCpuLowLink2TrainData: pd.DataFrame = highCpuLowLink2Data.sample(
        frac=0.8, random_state=0
    )
    highCpuHighLink2TrainData: pd.DataFrame = highCpuHighLink2Data.sample(
        frac=0.8, random_state=0
    )

    highCpuHighLinkTestData: pd.DataFrame = highCpuHighLinkData.drop(
        highCpuHighLinkTrainData.index
    )
    highCpuLowLinkTestData: pd.DataFrame = highCpuLowLinkData.drop(
        highCpuLowLinkTrainData.index
    )
    lowCpuHighLinkTestData: pd.DataFrame = lowCpuHighLinkData.drop(
        lowCpuHighLinkTrainData.index
    )
    lowCpuLowLinkTestData: pd.DataFrame = lowCpuLowLinkData.drop(
        lowCpuLowLinkTrainData.index
    )
    highCpuLowLink2TestData: pd.DataFrame = highCpuLowLink2Data.drop(
        highCpuLowLink2TrainData.index
    )
    highCpuHighLink2TestData: pd.DataFrame = highCpuHighLink2Data.drop(
        highCpuHighLink2TrainData.index
    )

    data: pd.DataFrame = pd.concat(
        [
            highCpuHighLinkData,
            highCpuLowLinkData,
            lowCpuHighLinkData,
            lowCpuLowLinkData,
            highCpuLowLink2Data,
            highCpuHighLink2Data,
        ]
    )

    plt.scatter(data["max_cpu"], data["link"], c=data[OUTPUT])
    plt.xlabel("CPU")
    plt.ylabel("Link")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{artifactsPath}/scatter.png")
    plt.clf()

    trainData: pd.DataFrame = pd.concat(
        [
            highCpuHighLinkTrainData,
            highCpuLowLinkTrainData,
            lowCpuHighLinkTrainData,
            lowCpuLowLinkTrainData,
            highCpuLowLink2TrainData,
            highCpuHighLink2TrainData,
        ]
    )
    testData: pd.DataFrame = pd.concat(
        [
            highCpuHighLinkTestData,
            highCpuLowLinkTestData,
            lowCpuHighLinkTestData,
            lowCpuLowLinkTestData,
            highCpuLowLink2TestData,
            highCpuHighLink2TestData,
        ]
    )

    templates: Any = (
        ydf.RandomForestLearner.hyperparameter_templates()
    )

    model = ydf.RandomForestLearner(
        label=OUTPUT, task=ydf.Task.REGRESSION, **templates["benchmark_rank1v1"]
    ).train(trainData, verbose=2)

    print(model.evaluate(testData))

    output: np.array = model.predict(testData)
    testData: pd.DataFrame = testData.assign(PredictedLatency=output.flatten())
    testData.to_csv(f"{artifactsPath}/predictions.csv", index=False)
    testData = testData.sort_values(by=OUTPUT).reset_index(drop=True)

    plt.scatter(testData.index, testData["PredictedLatency"], label="Predicted Latency")
    plt.scatter(testData.index, testData[OUTPUT], label="Actual q2")
    plt.xlabel("Index")
    plt.ylabel("Latency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{artifactsPath}/q2_vs_predicted_latency.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(testData[OUTPUT], testData["max_cpu"], testData["link"], c="blue", marker="P")
    ax.scatter(
        testData["PredictedLatency"],
        testData["max_cpu"],
        testData["link"],
        c="red",
        marker="*",
    )
    ax.set_xlabel("Latency")
    ax.set_ylabel("CPU")
    ax.set_zlabel('Latency')
    ax.grid(True)
    plt.savefig(f"{artifactsPath}/pred_scatter.png")
    plt.clf()

    fig, (plt1, plt2) = plt.subplots(1, 2)
    plt1.scatter(testData["max_cpu"], testData["link"], c=testData[OUTPUT])
    plt1.set_xlabel("CPU")
    plt1.set_ylabel("Link")
    plt1.grid(True)

    plt2.scatter(testData["max_cpu"], testData["link"], c=testData["PredictedLatency"])
    plt2.set_xlabel("CPU")
    plt2.set_ylabel("Link")
    plt2.grid(True)

    plt.savefig(f"{artifactsPath}/scatter_pred_valid.png")
    plt.clf()

    cpus: "list[float]" = [random.uniform(0, 2) for _ in range(1000000)]
    links: "list[float]" = [random.uniform(0, 4000) for _ in range(1000000)]
    data = pd.DataFrame({"max_cpu": cpus, "link": links})

    output = model.predict(data)
    data = data.assign(PredictedLatency=output.flatten())

    plt.scatter(data["max_cpu"], data["link"], c=data["PredictedLatency"])
    plt.xlabel("CPU")
    plt.ylabel("Link")
    plt.grid(True)
    plt.savefig(f"{artifactsPath}/scatter_sim.png")
    plt.clf()
