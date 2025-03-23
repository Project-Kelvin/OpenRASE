"""
This defines the surrogate model as a Bayesian Neural Network.
"""

import os
import random
from typing import Any
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.surrogacy.local_constants import (
    SURROGACY_PATH,
    SURROGATE_DATA_PATH,
    SURROGATE_MODELS_PATH,
)

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)

features: "list[str]" = ["link", "max_cpu"]
OUTPUT: str = "latency"
modelPath: str = SURROGATE_MODELS_PATH

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

highCpuHighLinkPath: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_hc_hl.csv",
)
highCpuLowLinkPath: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_hc_ll.csv",
)
lowCpuHighLinkPath: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_lc_hl.csv",
)
lowCpuLowLinkPath: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_lc_ll.csv",
)
highCpuLowLink2Path: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_hc_ll_2.csv",
)
highCpuHighLink2Path: str = os.path.join(
    SURROGATE_DATA_PATH,
    "latency_hc_hl_2.csv",
)


def train() -> None:
    """
    Trains the model.
    """

    artifactsPath: str = SURROGACY_PATH

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
        q1: float = data[OUTPUT].quantile(0.25)
        q3: float = data[OUTPUT].quantile(0.75)
        iqr: float = q3 - q1
        lowerBound: float = q1 - 1.5 * iqr
        upperBound: float = q3 + 1.5 * iqr

        return data[(data[OUTPUT] > lowerBound) & (data[OUTPUT] < upperBound)]

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

    filteredData: pd.DataFrame = pd.concat(
        [
            highCpuHighLinkData,
            highCpuLowLinkData,
            lowCpuHighLinkData,
            lowCpuLowLinkData,
            highCpuLowLink2Data,
            highCpuHighLink2Data,
        ]
    )

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 6)
    fig.set_dpi(300)
    points: Any = ax.scatter(
        filteredData["max_cpu"],
        filteredData["link"],
        c=filteredData[OUTPUT],
        vmin=0,
        vmax=2750,
        label="CPU vs. Link. Latency",
    )
    fig.colorbar(points, label="Latency(ms)")
    ax.set_xlabel("CPU")
    ax.set_ylabel("Link")
    ax.legend()
    ax.grid(True)
    plt.savefig(f"{artifactsPath}/cpu_link_latency_scatter.png")
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

    xTrain: np.ndarray = trainData[features].values
    yTrain: np.ndarray = trainData[OUTPUT].values
    xTest: np.ndarray = testData[features].values
    yTest: np.ndarray = testData[OUTPUT].values

    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(xTrain)
    activation: str = "sigmoid"

    model: tf.keras.Sequential = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(features),)),
            normalizer,
            tf.keras.layers.Dense(
                16, kernel_initializer="glorot_normal", activation=activation
            ),
            tf.keras.layers.Dense(
                16, kernel_initializer="glorot_normal", activation=activation
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.05),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    history: Any = model.fit(xTrain, yTrain, epochs=10000, validation_split=0.2)
    print(model.evaluate(xTest, yTest))

    plt.figure(1, (14, 6), dpi=300)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{artifactsPath}/loss_vs_val_loss.png")
    plt.clf()

    output: np.array = model.predict(xTest)
    testData: pd.DataFrame = testData.assign(PredictedLatency=output.flatten())
    testData.to_csv(f"{artifactsPath}/predictions.csv", index=False)
    testData = testData.sort_values(by=OUTPUT).reset_index(drop=True)

    plt.figure(2, (14, 6), dpi=300)
    plt.scatter(testData.index, testData["PredictedLatency"], label="Predicted Latency")
    plt.scatter(testData.index, testData[OUTPUT], label="Actual Latency")
    plt.xlabel("Index")
    plt.ylabel("Latency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{artifactsPath}/latency_vs_predicted_latency.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    fig.set_size_inches(14, 14)
    fig.set_dpi(300)
    ax.scatter(
        testData[OUTPUT],
        testData["max_cpu"],
        testData["link"],
        c="blue",
        marker="P",
        label="Actual Latency",
    )
    ax.scatter(
        testData["PredictedLatency"],
        testData["max_cpu"],
        testData["link"],
        c="red",
        marker="*",
        label="Predicted Latency",
    )
    ax.legend()
    ax.set_xlabel("Latency")
    ax.set_ylabel("CPU")
    ax.set_zlabel("Link")
    ax.grid(True)
    plt.savefig(f"{artifactsPath}/latency_vs_predicted_latency_scatter_3d.png")
    plt.clf()

    fig, (plt1, plt2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    fig.set_dpi(300)
    points: Any = plt1.scatter(
        testData["max_cpu"],
        testData["link"],
        c=testData[OUTPUT],
        label="Actual Latency",
        vmin=0,
        vmax=2750,
    )
    fig.colorbar(points, label="Latency(ms)")
    plt1.set_xlabel("CPU")
    plt1.set_ylabel("Link")
    plt1.grid(True)
    plt1.legend()

    points: Any = plt2.scatter(
        testData["max_cpu"],
        testData["link"],
        c=testData["PredictedLatency"],
        label="Predicted Latency",
        vmin=0,
        vmax=2750,
    )
    fig.colorbar(points, label="Latency(ms)")
    plt2.set_xlabel("CPU")
    plt2.set_ylabel("Link")
    plt2.grid(True)
    plt2.legend()

    plt.savefig(f"{artifactsPath}/latency_vs_predicted_latency_scatter_two_plots.png")
    plt.clf()

    cpus: "list[float]" = [random.uniform(0, 2) for _ in range(1000000)]
    links: "list[float]" = [random.uniform(0, 4000) for _ in range(1000000)]
    simData = pd.DataFrame({"max_cpu": cpus, "link": links})

    output = model.predict(simData[features].values)
    simData = simData.assign(PredictedLatency=output.flatten())

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 6)
    fig.set_dpi(300)
    points: Any = ax.scatter(
        simData["max_cpu"],
        simData["link"],
        c=simData["PredictedLatency"],
        label="Predicted Latency",
        vmin=0,
        vmax=2750,
    )
    fig.colorbar(points, label="Latency(ms)")
    ax.set_xlabel("CPU")
    ax.set_ylabel("Link")
    ax.grid(True)
    plt.savefig(f"{artifactsPath}/simulated_latency_scatter.png")
    plt.clf()

    model.save(f"{modelPath}/surrogate.keras")
