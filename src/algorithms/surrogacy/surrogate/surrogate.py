"""
This defines the surrogate model as a Bayesian Neural Network.
"""

import os
import random
from typing import Any
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt
import tf_keras

tf_keras.saving.get_custom_objects().clear()

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)


colsInclude: "list[str]" = ["link", "max_cpu"]
OUTPUT: str = "latency"
modelPath: str = getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/surrogate"
dataPath: str = (
    getConfig()["repoAbsolutePath"] + "/src/algorithms/surrogacy/data/latency.csv"
)

highCpuHighLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_hc_hl.csv"
)
highCpuLowLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_hc_ll.csv"
)
lowCpuHighLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_lc_hl.csv"
)
lowCpuLowLinkPath: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_lc_ll.csv"
)
highCpuLowLink2Path: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_hc_ll_2.csv"
)
highCpuHighLink2Path: str = os.path.join(
    getConfig()["repoAbsolutePath"], "src", "algorithms","surrogacy","data", "latency_hc_hl_2.csv"
)


@tf_keras.saving.register_keras_serializable(
    package="surrogate", name="negLogLikelihood"
)
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

    def _posteriorMeanField(
        self, kernelSize: int, biasSize: int = 0, dtype: Any = None
    ) -> tf_keras.Sequential:
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

        return tf_keras.Sequential(
            [
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Independent(
                        tfp.distributions.Normal(
                            loc=t[..., :n],
                            scale=1e-5 + tf.nn.softplus(0.005 + t[..., n:]),
                        ),
                        reinterpreted_batch_ndims=1,
                    )
                ),
            ]
        )

    def _priorTrainable(
        self, kernelSize: int, biasSize: int = 0, dtype=None
    ) -> tf_keras.Sequential:
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
        return tf_keras.Sequential(
            [
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Independent(
                        tfp.distributions.Normal(loc=t, scale=1),
                        reinterpreted_batch_ndims=1,
                    )
                ),
            ]
        )

    def train(self) -> None:
        """
        Trains the model.
        """

        highCpuHighLinkData: pd.DataFrame = pd.read_csv(highCpuHighLinkPath, sep=r"\s*,\s*")
        highCpuLowLinkData: pd.DataFrame = pd.read_csv(highCpuLowLinkPath, sep=r"\s*,\s*")
        lowCpuHighLinkData: pd.DataFrame = pd.read_csv(lowCpuHighLinkPath, sep=r"\s*,\s*")
        lowCpuLowLinkData: pd.DataFrame = pd.read_csv(lowCpuLowLinkPath, sep=r"\s*,\s*")
        highCpuLowLink2Data: pd.DataFrame = pd.read_csv(highCpuLowLink2Path, sep=r"\s*,\s*")
        highCpuHighLink2Data: pd.DataFrame = pd.read_csv(highCpuHighLink2Path, sep=r"\s*,\s*")

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
        highCpuHighLink2Data = highCpuHighLink2Data.groupby(["generation", "individual"]).agg(
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

        highCpuHighLinkTrainData: pd.DataFrame = highCpuHighLinkData.sample(frac=0.8, random_state=0)
        highCpuLowLinkTrainData: pd.DataFrame = highCpuLowLinkData.sample(frac=0.8, random_state=0)
        lowCpuHighLinkTrainData: pd.DataFrame = lowCpuHighLinkData.sample(frac=0.8, random_state=0)
        lowCpuLowLinkTrainData: pd.DataFrame = lowCpuLowLinkData.sample(frac=0.8, random_state=0)
        highCpuLowLink2TrainData: pd.DataFrame = highCpuLowLink2Data.sample(frac=0.8, random_state=0)
        highCpuHighLink2TrainData: pd.DataFrame = highCpuHighLink2Data.sample(frac=0.8, random_state=0)

        highCpuHighLinkTestData: pd.DataFrame = highCpuHighLinkData.drop(highCpuHighLinkTrainData.index)
        highCpuLowLinkTestData: pd.DataFrame = highCpuLowLinkData.drop(highCpuLowLinkTrainData.index)
        lowCpuHighLinkTestData: pd.DataFrame = lowCpuHighLinkData.drop(lowCpuHighLinkTrainData.index)
        lowCpuLowLinkTestData: pd.DataFrame = lowCpuLowLinkData.drop(lowCpuLowLinkTrainData.index)
        highCpuLowLink2TestData: pd.DataFrame = highCpuLowLink2Data.drop(highCpuLowLink2TrainData.index)
        highCpuHighLink2TestData: pd.DataFrame = highCpuHighLink2Data.drop(highCpuHighLink2TrainData.index)

        filteredData: pd.DataFrame = pd.concat(
            [highCpuHighLinkData, highCpuLowLinkData, lowCpuHighLinkData, lowCpuLowLinkData, highCpuLowLink2Data, highCpuHighLink2Data]
        )

        artifactsPath: str = (
            getConfig()["repoAbsolutePath"] + "/artifacts/experiments/surrogacy"
        )

        plt.scatter(filteredData["max_cpu"], filteredData["link"], c=filteredData[OUTPUT])
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

        history, model = self.trainModel(trainData)

        print(model.evaluate(testData[colsInclude].values, testData[OUTPUT].values))
        artifactsPath: str = (
            getConfig()["repoAbsolutePath"] + "/artifacts/experiments/surrogacy"
        )
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{artifactsPath}/plot.png")
        plt.clf()

        testData = self.predict(testData)
        testData.to_csv(f"{artifactsPath}/predictions.csv")

        plt.scatter(
            testData.index, testData["PredictedLatency"], label="Predicted Latency"
        )
        plt.scatter(testData.index, testData[OUTPUT], label="Actual q2")
        plt.xlabel("Index")
        plt.ylabel("Latency")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{artifactsPath}/q2_vs_predicted_latency.png")
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(
            testData[OUTPUT], testData["max_cpu"], testData["link"], c="blue", marker="P"
        )
        ax.scatter(
            testData["PredictedLatency"],
            testData["max_cpu"],
            testData["link"],
            c="red",
            marker="*",
        )
        ax.set_xlabel("Latency")
        ax.set_ylabel("CPU")
        ax.set_zlabel("Latency")
        ax.grid(True)
        plt.savefig(f"{artifactsPath}/pred_scatter_3d.png")
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

        cpus: "list[float]" = [random.uniform(0, 2) for _ in range(100)]
        links: "list[float]" = [random.uniform(0, 4000) for _ in range(100)]
        simData = pd.DataFrame({"max_cpu": cpus, "link": links, "latency": 0})

        simData = self.predict(simData)

        plt.scatter(simData["max_cpu"], simData["link"], c=simData["PredictedLatency"])
        plt.xlabel("CPU")
        plt.ylabel("Link")
        plt.grid(True)
        plt.savefig(f"{artifactsPath}/scatter_sim.png")
        plt.clf()

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(
            testData["max_cpu"],
            testData["link"],
            testData["Confidence"],
            c=testData["PredictedLatency"],
            marker="P",
        )
        ax.set_xlabel("CPU")
        ax.set_ylabel("Link")
        ax.set_zlabel("Confidence")
        ax.grid(True)
        plt.savefig(f"{artifactsPath}/sim_scatter_3d.png")
        plt.clf()

    def trainModel(self, data: pd.DataFrame, isUpdate: bool = False) -> "Tuple[Any, Any]":
        """
        Trains the model.

        Parameters:
            data (pd.DataFrame): the data to train on.
            isUpdate (bool): whether to update the model.

        Returns:
            Tuple[Any, Any]: the history and the model.
        """

        if isUpdate:
            with open(dataPath, "a", encoding="utf-8") as file:
                for row in data.iterrows():
                    file.write(f'{",".join(row)}\n')

        xTrain: np.ndarray = data[colsInclude].values
        yTrain: np.ndarray = data[OUTPUT].values

        activation: str = "relu"
        model: tf_keras.Sequential = tf_keras.Sequential(
            [
                tfp.layers.DenseVariational(
                    64,
                    make_posterior_fn=self._posteriorMeanField,
                    make_prior_fn=self._priorTrainable,
                    activation=activation,
                    kl_weight=1 / xTrain.shape[0],
                ),
                tfp.layers.DenseVariational(
                    64,
                    make_posterior_fn=self._posteriorMeanField,
                    make_prior_fn=self._priorTrainable,
                    activation=activation,
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

        epochs: int = 10000 if isUpdate else 10000
        model.compile(
            optimizer=tf_keras.optimizers.Adamax(learning_rate=0.1),
            loss=negLogLikelihood,
            metrics=[tf_keras.metrics.RootMeanSquaredError()],
        )
        history: Any = model.fit(xTrain, yTrain, epochs=epochs, validation_split=0.2)

        model.save(modelPath)
        self._model = model

        return history, model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the latency.

        Parameters:
            data (pd.DataFrame): the data frame to be predicted on.

        Returns:
            data (pd.DataFrame): the data frame with the prediction.
        """

        num: int = 1000

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
            mean: float = np.mean(predictions[i - num : i])
            std: float = np.std(predictions[i - num : i])

            means.append(mean)
            stds.append(std)

        outputData: pd.DataFrame = data.copy()
        outputData = outputData.assign(PredictedLatency=means, Confidence=stds)
        outputData = outputData.sort_values(by=OUTPUT).reset_index(drop=True)

        return outputData
