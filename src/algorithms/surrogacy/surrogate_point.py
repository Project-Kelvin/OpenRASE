"""
This defines the surrogate model as a Bayesian Neural Network.
"""

from typing import Any
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


colsExclude: "list[str]" = ["latency", "ar", "sfc", "individual", "generation", "memory", "avg_memory", "reqps", "total_hosts"]
colsInclude: "list[str]" = [
    "link",
    "no_sfcs",
    "avg_cpu",
    "avg_memory",
    "cpu",
    "memory"
]
def train() -> None:
    """
    Trains the model.
    """

    dataPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/data/latency.csv"
    data: pd.DataFrame = pd.read_csv(dataPath, sep=r'\s*,\s*')
    q1 = data["latency"].quantile(0.25)
    q3 = data["latency"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filteredData: pd.DataFrame = data[(data["latency"] > lower_bound) & (data["latency"] < upper_bound)]
    trainData: pd.DataFrame = filteredData.sample(frac=0.8, random_state=0)
    testData: pd.DataFrame = filteredData.drop(trainData.index)

    xTrain: np.ndarray = trainData[colsInclude].values
    yTrain: np.ndarray = trainData["latency"].values

    model: tf.keras.Sequential = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),
            tf.keras.layers.Dense(
                16,
                activation="relu",
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.025), loss="mae")
    history: Any = model.fit(xTrain, yTrain, epochs=2000, validation_split=0.2)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{'/'.join(dataPath.split('/')[:-2])}/plot.png")
    plt.clf()

    output: np.array = model.predict(testData[colsInclude].values)
    testData: pd.DataFrame = testData.assign(prediction=output.flatten())
    testData.to_csv("predictions.csv", index=False)

train()
