"""
This defines the surrogate model as a Bayesian Neural Network.
"""

import os
import pickle
import random
import time
from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from algorithms.hybrid.constants.surrogate import (
    SURROGACY_PATH,
    SURROGATE_DATA_PATH,
    SURROGATE_LOG_PATH,
    SURROGATE_MODELS_PATH,
)
from algorithms.hybrid.surrogate.combine_data import combineData

features: "list[str]" = ["max_link_score", "max_cpu"]
OUTPUT: str = "latency"
modelPath: str = SURROGATE_MODELS_PATH

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

MODEL_PATH: str = os.path.join(SURROGATE_MODELS_PATH, "surrogate.keras")


directory: str = SURROGACY_PATH
if not os.path.exists(directory):
    os.makedirs(directory)

surrogateDirectory: str = SURROGATE_LOG_PATH
if not os.path.exists(surrogateDirectory):
    os.makedirs(surrogateDirectory)

surrogateDataDirectory: str = SURROGATE_DATA_PATH
if not os.path.exists(surrogateDataDirectory):
    os.makedirs(surrogateDataDirectory)

surrogateFile: str = os.path.join(SURROGATE_LOG_PATH, "validation_error.log")

def train() -> None:
    """
    Trains the surrogate model.
    """

    trainingData, testData, allData = combineData()

    # Scatter plot of max_cpu vs max_link_score for all data
    plotData: pd.DataFrame = allData
    plt.figure(1, (8, 4), dpi=300)
    plt.scatter(
        plotData["max_cpu"],
        plotData["max_link_score"],
        c=plotData[OUTPUT],
        vmax=700,
    )
    plt.ylim(0, 1000)
    plt.xlabel("Maximum CPU Demand")
    plt.ylabel("Maximum Bandwidth Demand")
    plt.legend()
    plt.colorbar(label="Traffic Latency (ms)")
    plt.grid(True)
    plt.savefig(f"{SURROGACY_PATH}/cpu_vs_max_link.png")
    plt.clf()

    xTrain: np.ndarray = trainingData[features].values
    yTrain: np.ndarray = trainingData[OUTPUT].values
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
            tf.keras.layers.Dense(1, kernel_initializer="he_normal", activation="relu"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=0.05),
        loss="mse",
        metrics=["mape"],
    )
    history: Any = model.fit(xTrain, yTrain, epochs=75, verbose=1, validation_split=0.1)

    with open(surrogateFile, "a", encoding="utf8") as f:
        f.write(f"[{time.time()}] Validation Error: {model.evaluate(xTest, yTest)}\n")

    # rf = RandomForestRegressor(n_estimators=200)
    # rf.fit(xTrain, yTrain)

    # with open(surrogateFile, "a", encoding="utf8") as f:
    #     f.write(f"[{time.time()}] Validation Error: {rf.score(xTest, yTest)}\n")

    # Plotting the training history
    plt.figure(1, (8, 4), dpi=300)
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SURROGACY_PATH}/loss_vs_val_loss.png")
    plt.clf()

    output: np.ndarray = model.predict(xTest)
    testData: pd.DataFrame = testData.assign(PredictedLatency=output.flatten())
    # validationPrediction: np.ndarray = rf.predict(xTest)
    # testData: pd.DataFrame = testData.assign(PredictedLatency=validationPrediction.flatten())
    testData.to_csv(f"{SURROGACY_PATH}/predictions.csv", index=False)
    testData = testData.sort_values(by=OUTPUT).reset_index(drop=True)

    # Scatter plot of predicted vs actual latency
    plt.figure(2, (8, 4), dpi=300)
    plt.scatter(testData.index, testData["PredictedLatency"], label="Predicted Average Traffic Latency (ms)")
    plt.scatter(testData.index, testData[OUTPUT], label="Actual Average Traffic Latency (ms)")
    plt.xlabel("Index")
    plt.ylabel("Traffic Latency (ms)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{SURROGACY_PATH}/latency_vs_predicted_latency.png")
    plt.clf()

    # Scatter plot of max_cpu vs max_link_score with actual and predicted latency
    fig, (plt1, plt2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    fig.set_dpi(300)
    points: Any = plt1.scatter(
        testData["max_cpu"],
        testData["max_link_score"],
        c=testData[OUTPUT],
        label="Actual Traffic Latency (ms)",
        vmin=0,
        vmax=700,
    )
    plt1.set_yticks(np.arange(0, 700, 50))
    fig.colorbar(points, label="Traffic Latency (ms)")
    plt1.set_xlabel("Maximum CPU Demand")
    plt1.set_ylabel("Maximum Bandwidth Demand")
    plt1.grid(True)
    plt1.set_title("Actual Traffic Latency")

    points: Any = plt2.scatter(
        testData["max_cpu"],
        testData["max_link_score"],
        c=testData["PredictedLatency"],
        label="Predicted Traffic Latency (ms)",
        vmin=0,
        vmax=700,
    )
    fig.colorbar(points, label="Traffic Latency (ms)")
    plt2.set_yticks(np.arange(0, 700, 50))
    plt2.set_xlabel("Maximum CPU Demand")
    plt2.set_ylabel("Maximum Bandwidth Demand")
    plt2.grid(True)
    plt2.set_title("Predicted Traffic Latency")

    plt.savefig(f"{SURROGACY_PATH}/latency_vs_predicted_latency_scatter_two_plots.png")
    plt.clf()

    # cpus: "list[float]" = [random.uniform(0, 1.75) for _ in range(1000000)]
    # links: "list[float]" = [random.uniform(0, 300) for _ in range(1000000)]
    # simData = pd.DataFrame({"max_cpu": cpus, "max_link_score": links})

    # output = model.predict(simData[features].values)
    # simData = simData.assign(PredictedLatency=output.flatten())

    # # Scatter plot of max_cpu vs max_link_score for simulated data
    # fig, ax = plt.subplots()
    # fig.set_size_inches(6, 4)
    # fig.set_dpi(300)
    # points: Any = ax.scatter(
    #     simData["max_cpu"],
    #     simData["max_link_score"],
    #     c=simData["PredictedLatency"],
    #     vmin=0,
    #     vmax=700,
    # )
    # ax.set_yticks(np.arange(0, 300, 50))
    # fig.colorbar(points, label="Traffic Latency (ms)")
    # ax.set_xlabel("Maximum CPU Demand", fontsize=10)
    # ax.set_ylabel("Maximum Bandwidth Demand", fontsize=10)
    # ax.grid(True)
    # plt.savefig(f"{SURROGACY_PATH}/simulated_latency_scatter.png")
    # plt.clf()

    model.save(MODEL_PATH)
    # with open(MODEL_PATH, "wb") as f:
    #     pickle.dump(rf, f)



def predictLatency(data: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts the latency.

    Parameters:
        data (pd.DataFrame): the data.

    Returns:
        pd.DataFrame: the data with predicted latency.
    """

    model: tf.keras.Sequential = getSurrogateModel()
    output: np.ndarray = model.predict(data[features], verbose=0)
    data = data.assign(PredictedLatency=output.flatten())

    return data

def getSurrogateModel() -> tf.keras.Sequential:
    """
    Returns the model.

    Returns:
        tf.keras.Sequential: the model.
    """

    try:
        return tf.keras.models.load_model(MODEL_PATH)
        # with open(MODEL_PATH, "rb") as f:
        #     return pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")

        return None

def predictLatencyFromModel(data: np.ndarray, model: Any) -> np.ndarray:
    """
    Predicts the latency from the model.

    Parameters:
        data (np.ndarray): the data.
        model (Any): the model.

    Returns:
        np.ndarray: the predicted latency values.
    """

    output: np.ndarray = model.predict(data)

    return output.reshape(-1, 1)
