"""
This defines the surrogate model as a Bayesian Neural Network.
"""

import os
from typing import Any
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt
import tf_keras

tf_keras.saving.get_custom_objects().clear()

colsInclude: "list[str]" = [
    "link",
    "no_sfcs",
    "avg_cpu",
    "avg_memory",
    "cpu",
    "memory"
]

modelPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/surrogate"
dataPath: str = (
            getConfig()["repoAbsolutePath"]
            + "/src/algorithms/surrogacy/data/latency.csv"
        )

@tf_keras.saving.register_keras_serializable(package="surrogate", name="negLogLikelihood")
def negLogLikelihood(y, p_y) -> Any:
    """
    Returns the negative log likelihood.

    Parameters:
        y: the y value.
        p_y: the p_y value.

    Returns:
        Any: the negative log likelihood.
    """

    return -p_y.log_prob(y)

class Surrogate:
    """
    This defines the surrogate model.
    """

    def __init__(self) -> None:
        """
        Initializes the surrogate model.
        """

        if os.path.exists(modelPath):
            self._model: Any = tf_keras.models.load_model(modelPath, compile=False)

    def _posteriorMeanField(self, kernelSize: int, biasSize: int=0, dtype: Any=None) -> tf_keras.Sequential:
        """
        Defines the posterior mean field.

        Parameters:
            kernel_size: the kernel size.
            bias_size: the bias size.
            dtype: the data type.

        Returns:
            tf_keras.Sequential: the posterior mean field.
        """

        n = kernelSize + biasSize

        return tf_keras.Sequential([
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
                tfp.distributions.Normal(loc=t[..., :n],
                            scale=1e-5 + tf.nn.softplus(0.005 + t[..., n:])),
                reinterpreted_batch_ndims=1)),
        ])

    def _priorTrainable(self, kernelSize: int, biasSize: int=0, dtype=None) -> tf_keras.Sequential:
        """
        Defines the prior trainable.

        Parameters:
            kernel_size: the kernel size.
            bias_size: the bias size.
            dtype: the data type.

        Returns:
            tf_keras.Sequential: the prior trainable.
        """

        n = kernelSize + biasSize
        return tf_keras.Sequential([
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
                tfp.distributions.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ])

    def train(self) -> None:
        """
        Trains the model.
        """

        data: pd.DataFrame = pd.read_csv(dataPath, sep=r"\s*,\s*")
        q1 = data["latency"].quantile(0.25)
        q3 = data["latency"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filteredData: pd.DataFrame = data[
            (data["latency"] > lower_bound) & (data["latency"] < upper_bound)
        ]

        trainData: pd.DataFrame = filteredData.sample(frac=0.8, random_state=0)
        testData: pd.DataFrame = filteredData.drop(trainData.index)

        history: Any = self.trainModel(trainData)

        artifactsPath: str = getConfig()["repoAbsolutePath"] + "/artifacts/experiments/surrogacy"
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{artifactsPath}/plot.png")
        plt.clf()

        output: pd.DataFrame = self.predict(testData)
        output.to_csv(f"{artifactsPath}/predictions.csv")

    def trainModel(self, data: pd.DataFrame, shouldWrite: bool = False) -> Any:
        """
        Trains the model.

        Parameters:
            data (pd.DataFrame): the data to train on.
            shouldWrite (bool): whether to write the data.

        Returns:
            Any: the history.
        """

        if shouldWrite:
            with open(dataPath, "a+", encoding="utf-8") as file:
                for row in data.iterrows():
                    file.write(f'{",".join(row)}\n')

        xTrain: np.ndarray = data[colsInclude].values
        yTrain: np.ndarray = data["latency"].values

        model: tf_keras.Sequential = tf_keras.Sequential(
            [
                tfp.layers.DenseVariational(
                    16,
                    make_posterior_fn=self._posteriorMeanField,
                    make_prior_fn=self._priorTrainable,
                    activation="relu",
                    kl_weight=1 / xTrain.shape[0],
                ),
                tfp.layers.DenseVariational(
                    16,
                    make_posterior_fn=self._posteriorMeanField,
                    make_prior_fn=self._priorTrainable,
                    activation="relu",
                    kl_weight=1 / xTrain.shape[0],
                ),
                tfp.layers.DenseVariational(
                    2,
                    make_posterior_fn=self._posteriorMeanField,
                    make_prior_fn=self._priorTrainable,
                    kl_weight=1 / xTrain.shape[0],
                ),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(
                        loc=t[..., :1],
                        scale=1e-3 + tf.math.softplus(0.005 * t[..., 1:]),
                    )
                ),
            ]
        )

        model.compile(optimizer=tf_keras.optimizers.Adamax(learning_rate=0.025), loss=negLogLikelihood)
        history: Any = model.fit(xTrain, yTrain, epochs=3000, validation_split=0.2)

        model.save(modelPath)
        self._model = model

        return history

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the latency.

        Parameters:
            data (pd.DataFrame): the data frame to be predicted on.

        Returns:
            data (pd.DataFrame): the data frame with the prediction.
        """

        num: int = 100

        predictions: "list[list[float]]" = []
        dataArray: "list[list[float]]" = []

        testData: pd.DataFrame = data[colsInclude]

        for _index, row in testData.iterrows():
            dataRow: "list[float]" = row.values
            for _i in range(num):
                dataArray.append(np.asarray(dataRow).astype("float32"))

        predictions.extend(self._model.predict(np.array(dataArray)))
        means: "list[float]" = []
        stds: "list[float]" = []

        for i in range(num, len(predictions) + num, num):
            mean: float = np.mean(predictions[i-num:i])
            std: float = np.std(predictions[i-num:i])

            means.append(mean)
            stds.append(std)

        outputData: pd.DataFrame = data.copy()
        outputData = outputData.assign(PredictedLatency = means, Confidence = stds)

        return outputData
