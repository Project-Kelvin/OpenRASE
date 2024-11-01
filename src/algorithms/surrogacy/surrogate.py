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

def getHostScores(reqps: int, topology: Topology, egs: "list[EmbeddingGraph]", embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]" ) -> "dict[str, ResourceDemand]":
    """
    Gets the host scores.

    Parameters:
        reqps (int): the reqps.
        topology (Topology): the topology.
        egs (list[EmbeddingGraph]): the Embedding Graphs.
        embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.

    Returns:
        dict[str, ResourceDemand]: the host scores.
    """

    dataToCache: "dict[str, list[float]]" = {}
    def parseEG(vnf: VNF, _depth: int) -> None:
        """
        Parses an EG.

        Parameters:
            vnf (VNF): the VNF.
            _depth (int): the depth.
            egID (str): the EG ID.
        """

        nonlocal dataToCache

        dataToCache[vnf["vnf"]["id"]] = [reqps]

    for eg in egs:
        traverseVNF(eg["vnfs"], parseEG, shouldParseTerminal=False)

    calibrate = Calibrate()
    calibrate.predictAndCache(dataToCache)

    hostResourceData: "dict[str, ResourceDemand]" = {}
    for host, sfcs  in embeddingData.items():
        otherCPU: float = 0
        otherMemory: float = 0

        for vnfs in sfcs.values():
            for vnf, depth in vnfs:
                divisor: int = 2**(depth-1)
                effectiveReqps: float = reqps / divisor
                demands: ResourceDemand = calibrate.getVNFResourceDemandForReqps(vnf, effectiveReqps)

                vnfCPU: float = demands["cpu"]
                vnfMemory: float = demands["memory"]
                otherCPU += vnfCPU
                otherMemory += vnfMemory

        hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)

    for host, data in hostResourceData.items():
        topoHost: Host = [h for h in topology["hosts"] if h["id"] == host][0]
        hostCPU: float = topoHost["cpu"]
        hostMemory: float = topoHost["memory"]
        data["cpu"] = data["cpu"] / hostCPU
        data["memory"] = data["memory"] / hostMemory

    return hostResourceData

def getLinkScores(reqps: int, topology: Topology, egs: "list[EmbeddingGraph]", linkData: "dict[str, dict[str, float]]") -> "dict[str, float]":
    """
    Gets the link scores.

    Parameters:
        reqps (int): the reqps.
        topology (Topology): the topology.
        egs (list[EmbeddingGraph]): the Embedding Graphs.
        linkData (dict[str, dict[str, float]]): the link data.

    Returns:
        dict[str, float]: the link scores.
    """

    linkScores: "dict[str, float]" = {}
    for eg in egs:
        totalLinkScore: float = 0
        for egLink in eg["links"]:
            links: "list[str]" = [egLink["source"]["id"]]
            links.extend(egLink["links"])
            links.append(egLink["destination"]["id"])
            divisor: int = egLink["divisor"]
            reqps: float = reqps / divisor

            for linkIndex in range(len(links) - 1):
                source: str = links[linkIndex]
                destination: str = links[linkIndex + 1]

                totalRequests: int = 0

                if f"{source}-{destination}" in linkData:
                    for key, data in linkData[f"{source}-{destination}"].items():
                        totalRequests += data * reqps
                elif f"{destination}-{source}" in linkData:
                    for data in linkData[f"{destination}-{source}"].values():
                        totalRequests += data * reqps

                bandwidth: float = [link["bandwidth"] for link in topology["links"] if (link["source"] == source and link["destination"] == destination) or (link["source"] == destination and link["destination"] == source)][0]

                linkScore: float = getLinkScore(reqps, totalRequests, bandwidth)

                totalLinkScore += linkScore

        linkScores[eg["sfcID"]] = totalLinkScore

    return linkScores

def getSFCScores(data: "list[dict[str, dict[str, Union[int, float]]]]", topology: Topology, egs: "list[EmbeddingGraph]", embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]", linkData: "dict[str, dict[str, float]]" ) -> "list[list[Union[str, float]]]":
    """
    Gets the SFC scores.

    Parameters:
        data (list[dict[str, dict[str, Union[int, float]]]]): the data.
        topology (Topology): the topology.
        egs (list[EmbeddingGraph]): the Embedding Graphs.
        embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): the embedding data.
        linkData (dict[str, dict[str, float]]): the link data.

    Returns:
        list[list[Union[str, float]]]: the SFC scores.
    """

    vnfsInEGs: "dict[str, set[str]]" = {}
    def parseEG(vnf: VNF, _depth: int, egID: str) -> None:
        """
        Parses an EG.

        Parameters:
            vnf (VNF): the VNF.
            _depth (int): the depth.
            egID (str): the EG ID.
        """

        nonlocal vnfsInEGs

        if egID not in vnfsInEGs:
            vnfsInEGs[egID] = {vnf["vnf"]["id"]}
        else:
            vnfsInEGs[egID].add(vnf["vnf"]["id"])

    for eg in egs:
        traverseVNF(eg["vnfs"], parseEG, eg["sfcID"], shouldParseTerminal=False)

    dataToCache: "dict[str, list[float]]" = {}

    for step in data:
        for sfc, sfcData in step.items():
            for vnf in vnfsInEGs[sfc]:
                if vnf in dataToCache:
                    dataToCache[vnf].append(sfcData["reqps"])
                else:
                    dataToCache[vnf] = [sfcData["reqps"]]

    calibrate = Calibrate()
    calibrate.predictAndCache(dataToCache)
    rows: "list[list[Union[str, float]]]" = []
    for step in data:
        """ hostResourceData: "dict[str, ResourceDemand]" = {}
        for host, sfcs  in embeddingData.items():
            otherCPU: float = 0
            otherMemory: float = 0

            for sfc, vnfs in sfcs.items():
                for vnf, depth in vnfs:
                    divisor: int = 2**(depth-1)
                    reqps: float = (step[sfc]["reqps"] if sfc in step else 0) / divisor
                    demands: ResourceDemand = calibrate.getVNFResourceDemandForReqps(vnf, reqps)

                    vnfCPU: float = demands["cpu"]
                    vnfMemory: float = demands["memory"]
                    otherCPU += vnfCPU
                    otherMemory += vnfMemory

            hostResourceData[host] = ResourceDemand(cpu=otherCPU, memory=otherMemory)
        TUI.appendToSolverLog("Resource consumption of hosts calculated.") """

        hostVNFs: "dict[str, int]" = {}
        for host, sfcs in embeddingData.items():
            hostVNFs[host] = sum([len(vnfs) for vnfs in sfcs.values()])

        for sfc, sfcData in step.items():
            totalCPUScore: float = 0
            totalMemoryScore: float = 0
            totalLinkScore: float = 0
            eg: EmbeddingGraph = [graph for graph in egs if graph["sfcID"] == sfc][0]
            row: "list[Union[str, float]]" = []


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
                reqps: float = sfcData["reqps"] / divisor
                demands: ResourceDemand = calibrate.getVNFResourceDemandForReqps(vnf["vnf"]["id"], reqps)

                vnfCPU: float = demands["cpu"]
                vnfMemory: float = demands["memory"]

                host: Host = [host for host in topology["hosts"] if host["id"] == vnf["host"]["id"]][0]
                hostCPU: float = host["cpu"]
                hostMemory: float = host["memory"]

                cpuScore: float = getScore(vnfCPU, hostVNFs[vnf["host"]["id"]], hostCPU)
                memoryScore: float = getScore(vnfMemory, hostVNFs[vnf["host"]["id"]], hostMemory)
                totalCPUScore += cpuScore
                totalMemoryScore += memoryScore

            traverseVNF(eg["vnfs"], parseVNF, shouldParseTerminal=False)

            TUI.appendToSolverLog(f"CPU Score: {totalCPUScore}. Memory Score: {totalMemoryScore}.")
            for egLink in eg["links"]:
                links: "list[str]" = [egLink["source"]["id"]]
                links.extend(egLink["links"])
                links.append(egLink["destination"]["id"])
                divisor: int = egLink["divisor"]
                reqps: float = sfcData["reqps"] / divisor

                for linkIndex in range(len(links) - 1):
                    source: str = links[linkIndex]
                    destination: str = links[linkIndex + 1]

                    totalRequests: int = 0

                    if f"{source}-{destination}" in linkData:
                        for key, data in linkData[f"{source}-{destination}"].items():
                            totalRequests += data * (step[key]["reqps"] if key in step else 0)
                    elif f"{destination}-{source}" in linkData:
                        for data in linkData[f"{destination}-{source}"].values():
                            totalRequests += data * (step[key]["reqps"] if key in step else 0)

                    bandwidth: float = [link["bandwidth"] for link in topology["links"] if (link["source"] == source and link["destination"] == destination) or (link["source"] == destination and link["destination"] == source)][0]

                    linkScore: float = getLinkScore(reqps, totalRequests, bandwidth)

                    totalLinkScore += linkScore

            TUI.appendToSolverLog(f"Link Score: {totalLinkScore}.")
            row.append(sfc)
            row.append(sfcData["reqps"])
            row.append(totalCPUScore)
            row.append(totalMemoryScore)
            row.append(totalLinkScore)
            row.append(sfcData["latency"])
            rows.append(row)

    return rows

def getScore(demand: float, totalVNFs: int, resource: float) -> float:
    """
    Gets the resource score.

    Parameters:
        demand (float): the demand.
        totalVNFs (int): the total VNFs.
        resource (float): the resource.

    Returns:
        float: the score.
    """

    return demand / (resource / totalVNFs)

def getLinkScore(demand: float, totalDemand: int, resource: float) -> float:
    """
    Gets the resource score.

    Parameters:
        demand (float): the demand.
        totalDemand (int): the total demand.
        resource (float): the resource.

    Returns:
        float: the score.
    """

    return (demand / totalDemand) * resource
