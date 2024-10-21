"""
This defines the surrogate model as a Bayesian Neural Network.
"""

from typing import Any, Callable, Tuple, Union

from shared.models.embedding_graph import VNF, EmbeddingGraph
from shared.models.topology import Host, Topology
from calibrate.calibrate import Calibrate
from models.calibrate import ResourceDemand
import numpy as np
from shared.utils.config import getConfig
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
import matplotlib.pyplot as plt
import tf_keras
from utils.tui import TUI

from utils.embedding_graph import traverseVNF


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
    filteredData: pd.DataFrame = data.loc[(data["latency"] != 10000) &
                                    (data["latency"] != 20000) &
                                    (data["latency"] != 30000) &
                                    (data["latency"] != 40000) &
                                    (data["latency"] != 50000)]
    trainData: pd.DataFrame = filteredData.sample(frac=0.8, random_state=0)
    testData: pd.DataFrame = filteredData.drop(trainData.index)

    xTrain: np.ndarray = trainData[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
    yTrain: np.ndarray = trainData["latency"].values
    _xTest: np.ndarray = testData[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
    _yTest: np.ndarray = testData["latency"].values

    negLogLikelihood: "Callable[[tf.Tensor, tf.Tensor], tf.Tensor]" = lambda y, p_y: -p_y.log_prob(y)
    model: tf_keras.Sequential = tf_keras.Sequential([
        tfp.layers.DenseVariational(16, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, activation="relu", kl_weight=1/xTrain.shape[0]),
        tfp.layers.DenseVariational(8, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, activation="relu", kl_weight=1/xTrain.shape[0]),
        tfp.layers.DenseVariational(4, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, activation="relu", kl_weight=1/xTrain.shape[0]),
        tfp.layers.DenseVariational(2, make_posterior_fn=posteriorMeanField, make_prior_fn=priorTrainable, kl_weight=1/xTrain.shape[0]),
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.Normal(loc=t[..., :1],
                                                scale=1e-3 + tf.math.softplus(0.005 * t[..., 1:]))
        )
    ])

    model.compile(optimizer=tf_keras.optimizers.Adam(learning_rate=0.05), loss=negLogLikelihood)
    history: Any = model.fit(xTrain, yTrain, epochs=30000, validation_split=0.2)

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{'/'.join(dataPath.split('/')[:-2])}/plot.png")
    plt.clf()

    for _index, row in testData.iterrows():
        w: "list[float]" = row[["w1", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9"]].values
        mean, std = predict(w, model)
        print(f"Predicted: {mean}, Actual: {row['latency']}, Standard Deviation: {std}")


def predict(w: "list[float]", model) -> "Tuple[float, float]":
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


def getSFCScoresForEGs(trafficModel: "list[dict[str, int]]", topology: Topology, egs: "list[EmbeddingGraph]", embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]", linkData: "dict[str, dict[str, float]]") -> "list[list[Union[str, float]]]":
    """
    Gets the SFC scores.

    Parameters:
        trafficModel (list[dict[str, int]]): the traffic model.
        egs (list[EmbeddingGraph]): the Embedding Graphs.
        topology (Topology): the topology.
        embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
        linkData (dict[str, dict[str, float]]): the link data.

    Returns:
        list[list[Union[str, float]]]: the SFC scores.
    """

    rows: "list[list[Union[str, float]]]" =[]

    for eg in egs:
        rows.extend(getSFCScoresForTrafficModel(trafficModel, topology, eg, embeddingData, linkData))

def getSFCScoresForTrafficModel(trafficModel: "list[dict[str, int]]", topology: Topology, eg: EmbeddingGraph, embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]", linkData: "dict[str, dict[str, float]]") -> "list[list[Union[str, float]]]":
    """
    Gets the SFC scores.

    Parameters:
        trafficModel (list[dict[str, int]]): the traffic model.
        topology (Topology): the topology.
        eg (EmbeddingGraph): the Embedding Graph.
        embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
        linkData (dict[str, dict[str, float]]): the link data.

    Returns:
        list[list[Union[str, float]]]: the SFC scores.
    """

    return [getSFCScore(reqps, topology, eg, embeddingData, linkData) for reqps in trafficModel]

def getSFCScore(reqps: "dict[str, int]", topology: Topology, eg: EmbeddingGraph, embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]", linkData: "dict[str, dict[str, float]]" ) -> "list[Union[str, float]]":
    """
    Gets the SFC scores.

    Parameters:
        reqps (dict[str, int]) : the requests per second by SFC.
        topology (Topology): the topology.
        eg (EmbeddingGraph): the Embedding Graph.
        embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
        linkData (dict[str, dict[str, float]]): the link data.

    Returns:
        list[Union[str, float]]: the SFC scores.
    """

    calibrate = Calibrate()
    hostResourceData: "dict[str, ResourceDemand]" = {}

    for host, sfcs  in embeddingData.items():
        otherCPU: float = 0
        otherMemory: float = 0

        for sfc, vnfs in sfcs.items():
            for vnf, depth in vnfs:
                divisor: int = 2**(depth-1)
                demands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(reqps[sfc] if sfc in reqps else 0 /divisor)

                vnfCPU: float = demands[vnf]["cpu"]
                vnfMemory: float = demands[vnf]["memory"]
                otherCPU += vnfCPU
                otherMemory += vnfMemory

        hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)

    TUI.appendToSolverLog(f"Resource consumption of hosts calculated.")
    row: "list[float]" = []
    totalCPUScore: float = 0
    totalMemoryScore: float = 0
    totalLinkScore: float = 0

    def parseVNF(vnf: VNF, depth: int) -> None:
        """
        Parses a VNF.

        Parameters:
            vnf (VNF): the VNF.
            depth (int): the depth.
        """

        nonlocal totalCPUScore
        nonlocal totalMemoryScore

        divisor: int = 2**(depth-1)
        demands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(reqps[eg["sfcID"]] if eg["sfcID"] in reqps else 0 /divisor)

        vnfCPU: float = demands[vnf["vnf"]["id"]]["cpu"]
        vnfMemory: float = demands[vnf["vnf"]["id"]]["memory"]

        host: Host = [host for host in topology["hosts"] if host["id"] == vnf["host"]["id"]][0]
        hostCPU: float = host["cpu"]
        hostMemory: float = host["memory"]

        cpuScore: float = getCPUScore(vnfCPU, hostResourceData[vnf["host"]["id"]]["cpu"], hostCPU)
        memoryScore: float = getMemoryScore(vnfMemory, hostResourceData[vnf["host"]["id"]]["memory"], hostMemory)

        totalCPUScore += cpuScore
        totalMemoryScore += memoryScore

    traverseVNF(eg["vnfs"], parseVNF, shouldParseTerminal=False)

    TUI.appendToSolverLog(f"CPU Score: {totalCPUScore}. Memory Score: {totalMemoryScore}.")
    for egLink in eg["links"]:
        links: "list[str]" = [egLink["source"]["id"]]
        links.extend(egLink["links"])
        links.append(egLink["destination"]["id"])
        divisor: int = egLink["divisor"]

        for linkIndex in range(len(links) - 1):
            source: str = links[linkIndex]
            destination: str = links[linkIndex + 1]

            totalRequests: int = 0

            if f"{source}-{destination}" in linkData:
                for sfc, requests in reqps.items():
                    totalRequests += linkData[f"{source}-{destination}"][sfc]
            elif f"{destination}-{source}" in linkData:
                for sfc, requests in reqps.items():
                    totalRequests += linkData[f"{destination}-{source}"][sfc]

            bandwidth: float = [link["bandwidth"] for link in topology["links"] if (link["source"] == source and link["destination"] == destination) or (link["source"] == destination and link["destination"] == source)][0]

            linkScore: float = getLinkScore(reqps[eg["sfcID"]] if eg["sfcID"] in reqps else 0 /divisor, totalRequests, bandwidth)

            totalLinkScore += linkScore

    TUI.appendToSolverLog(f"Link Score: {totalLinkScore}.")
    row.append(eg["sfcID"])
    row.append(reqps[eg["sfcID"]])
    row.append(totalCPUScore)
    row.append(totalMemoryScore)
    row.append(totalLinkScore)

    return row

def getCPUScore(cpuDemand: float, totalCPUDemand: float, hostCPU: float) -> float:
    """
    Gets the CPU score.

    Parameters:
        cpuDemand (float): the CPU demand.
        totalCPUDemand (float): the total CPU demand.
        hostCPU (float): the host CPU.

    Returns:
        float: the CPU score.
    """

    return ((cpuDemand / totalCPUDemand) * hostCPU) ** -1

def getMemoryScore(memoryDemand: float, totalMemoryDemand: float, hostMemory: float) -> float:
    """
    Gets the memory score.

    Parameters:
        memoryDemand (float): the memory demand.
        totalMemoryDemand (float): the total memory demand.
        hostMemory (float): the host memory.

    Returns:
        float: the memory score.
    """

    return ((memoryDemand / totalMemoryDemand) * hostMemory) ** -1

def getLinkScore(requests: int, totalRequests: int, bandwidth: float) -> float:
    """
    Gets the link score.

    Parameters:
        requests (int): the requests.
        totalRequests (int): the total requests.
        bandwidth (float): the bandwidth.

    Returns:
        float: the link score.
    """

    return ((requests / totalRequests) * bandwidth) ** -1
