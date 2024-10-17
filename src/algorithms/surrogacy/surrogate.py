"""
This defines the surrogate model as a Bayesian Neural Network.
"""

from typing import Any, Callable, Tuple
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt
import tf_keras

dataPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/data/weights.csv"
data: pd.DataFrame = pd.read_csv(dataPath, sep=r'\s*,\s*')
filteredData: pd.DataFrame = data.loc[(data["latency"] != 10000) &
                                  (data["latency"] != 20000) &
                                  (data["latency"] != 30000) &
                                  (data["latency"] != 40000) &
                                  (data["latency"] != 50000)]
trainData: pd.DataFrame = filteredData.sample(frac=0.8, random_state=0)
testData: pd.DataFrame = filteredData.drop(trainData.index)

xTrain: np.ndarray = trainData[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
yTrain: np.ndarray = trainData["latency"].values
xTest: np.ndarray = testData[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
yTest: np.ndarray = testData["latency"].values

negLogLikelihood: "Callable[[tf.Tensor, tf.Tensor], tf.Tensor]" = lambda y, p_y: -p_y.log_prob(y)

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
                        scale=1e-5 + tf.nn.softplus(0.05 + t[..., n:])),
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

model: tf_keras.Sequential = tf_keras.Sequential([
    tfp.layers.DenseVariational(1, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, activation="relu", kl_weight=1/xTrain.shape[0]),
    tfp.layers.DenseVariational(2, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, kl_weight=1/xTrain.shape[0]),
    tfp.layers.DistributionLambda(
        lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                            scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))
    )
])

model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.05), loss=negLogLikelihood)
history: Any = model.fit(xTrain, yTrain, epochs=2000, validation_split=0.2)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(f"{'/'.join(dataPath.split('/')[:-2])}/plot.png")
plt.clf()


def predict(w: "list[float]") -> "Tuple[float, float]":
    """
    Predicts the latency.

    Parameters:
        w (list[float]): the weights.

    Returns:
        float: the latency and the standard deviation.
    """

    num: int = 100

    predictions: "list[float]" = []
    for _i in range(num):
        predictions.append(model.predict(np.array([w]))[0][0])

    return np.median(predictions), np.quantile(predictions, 0.75) - np.quantile(predictions, 0.25)

for index, row in testData.iterrows():
    w: "list[float]" = row[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
    mean, std = predict(w)
    print(f"Predicted: {mean}, Actual: {row['latency']}, Standard Deviation: {std}")
