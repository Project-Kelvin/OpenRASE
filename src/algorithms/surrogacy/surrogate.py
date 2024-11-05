"""
This defines the surrogate model as a Bayesian Neural Network.
"""

from typing import Any, Callable
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt
import tf_keras


def posteriorMeanField(kernelSize: int, biasSize: int=0, dtype: Any=None) -> tf_keras.Sequential:
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

def priorTrainable(kernelSize: int, biasSize: int=0, dtype=None) -> tf_keras.Sequential:
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

def train() -> None:
    """
    Trains the model.
    """

    dataPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/data/weights.csv"
    data: pd.DataFrame = pd.read_csv(dataPath, sep=r'\s*,\s*')
    q1 = data["latency"].quantile(0.25)
    q3 = data["latency"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filteredData: pd.DataFrame = data[(data["latency"] > lower_bound) & (data["latency"] < upper_bound)]
    trainData: pd.DataFrame = filteredData.sample(frac=0.9, random_state=0)
    testData: pd.DataFrame = filteredData.drop(trainData.index)

    xTrain: np.ndarray = trainData[["cpu", "memory", "link"]].values
    yTrain: np.ndarray = trainData["latency"].values

    negLogLikelihood: "Callable[[tf.Tensor, tf.Tensor], tf.Tensor]" = lambda y, p_y: -p_y.log_prob(y)
    model: tf_keras.Sequential = tf_keras.Sequential([
        tfp.layers.DenseVariational(2, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, activation="relu", kl_weight=1/xTrain.shape[0]),
        tfp.layers.DenseVariational(2, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, kl_weight=1/xTrain.shape[0]),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                                scale=1e-3 + tf.math.softplus(0.005 * t[..., 1:]))
        )
    ])

    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.05), loss=negLogLikelihood)
    history: Any = model.fit(xTrain, yTrain, epochs=100, validation_split=0.2)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{'/'.join(dataPath.split('/')[:-2])}/plot.png")
    plt.clf()

    output: pd.DataFrame = predict(testData, model)
    output.to_csv("predictions.csv")


def predict(data: pd.DataFrame, model) -> pd.DataFrame:
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

    for _index, row in data.iterrows():
        dataRow: "list[float]" = row[["cpu", "memory", "link"]].values

        for _i in range(num):
            dataArray.append(np.asarray(dataRow).astype("float32"))

    predictions.extend(model.predict(np.array(dataArray)))
    means: "list[float]" = []
    stds: "list[float]" = []

    for i in range(num, len(predictions) + num, num):
        median: float = np.mean(predictions[i-num:i])
        std: float = np.std(predictions[i-num:i])

        means.append(median)
        stds.append(std)

    outputData: pd.DataFrame = data.copy()
    outputData = outputData.assign(PredictedLatency = means, Confidence = stds)


    return outputData
