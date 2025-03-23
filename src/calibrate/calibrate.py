"""
This code is used to calibrate the CPU, memory, and bandwidth demands of VNFs.
"""

from copy import deepcopy
import os
import random
from time import sleep
from timeit import default_timer
import json
from typing import Any
from shared.models.embedding_graph import VNF
from shared.constants.embedding_graph import TERMINAL
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from calibrate.benchmarking_models import vnfModels
from mano.telemetry import Telemetry
from models.calibrate import ResourceDemand
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.data import hostDataToFrame, mergeHostAndTrafficData
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI
from constants.topology import SERVER, SFCC

EPOCHS: int = 200

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)


class Calibrate:
    """
    Class that calibrates the VNFs.
    """

    def __init__(self) -> None:
        """
        Constructor for the class.

        Parameters:
            trafficDesign (dict): The design of the traffic generator.
        """

        self._config: Config = getConfig()
        self._calDir: str = f"{self._config['repoAbsolutePath']}/artifacts/calibrations"
        self._modelName: str = "model.keras"
        self._headers: "list[str]" = [
            "cpu",
            "memory",
            "median",
            "reqps",
        ]
        self._cache: "dict[str, dict[str, ResourceDemand]]" = {}
        self._models: "dict[str, dict[str, Any]]" = {}

        if not os.path.exists(
            f"{self._config['repoAbsolutePath']}/artifacts/calibrations"
        ):
            os.makedirs(f"{self._config['repoAbsolutePath']}/artifacts/calibrations")

    def _trainModel(self, metric: str, vnf: str, epochs: int = EPOCHS) -> float:
        """
        Train a model to predict the metric from the data in the file.

        Parameters:
            metric (str): The metric to predict.
            vnf (str): The VNF to predict the metric for.
            epochs (int): The number of epochs to train the model for.#

        Returns:
            float: The test loss.
        """

        directory: str = (
            f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
        )
        file = f"{directory}/calibration_data.csv"

        directory: str = f"{self._calDir}/{vnf}"

        data: DataFrame = pd.read_csv(file)
        data["memory"] = data["memory"] / (1024 * 1024)
        data = data[[metric, "reqps"]]

        data = data[(data[metric] != 0) & (data["reqps"] != 0)]

        if metric == self._headers[0]:
            q1: float = data[metric].quantile(0.25)
            q3: float = data[metric].quantile(0.75)
            iqr: float = q3 - q1
            lowerBound: float = q1 - (1.5 * iqr)
            upperBound: float = q3 + (1.5 * iqr)
            data = data[(data[metric] <= upperBound) & (data[metric] >= lowerBound)]

        trainData: DataFrame = data.sample(frac=0.8, random_state=0)
        testData: DataFrame = data.drop(trainData.index)

        trainFeatures: DataFrame = trainData.copy()
        testFeatures: DataFrame = testData.copy()

        trainLabels: Series = trainFeatures.pop(metric)
        testLabels: Series = testFeatures.pop(metric)

        reqps: Any = np.array(trainFeatures["reqps"])
        normalizer: Any = tf.keras.layers.Normalization(
            input_shape=[
                1,
            ],
            axis=None,
        )
        normalizer.adapt(reqps)

        model: Any = tf.keras.Sequential(
            [
                normalizer,
                *vnfModels[vnf][metric],
                tf.keras.layers.Dense(units=1),
            ]
        )

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.025), loss="mse")

        print("Training the model.")
        history: Any = model.fit(
            trainFeatures["reqps"],
            trainLabels,
            epochs=epochs,
            verbose=0,
            validation_split=0.2,
        )
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel(f"Error [{metric}]")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{directory}/{metric}_loss.png")
        plt.clf()

        print("Evaluating the model.")
        testResult: Any = model.evaluate(
            testFeatures["reqps"], testLabels, verbose=0
        )

        print(f"Test loss: {testResult}")

        x = tf.linspace(0, 500, 501)
        y = model.predict(x)

        plt.scatter(trainFeatures["reqps"], trainLabels, label="Data")
        plt.plot(x, y, color="k", label="Predictions")
        plt.xlabel("reqps")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"{directory}/{metric}_trend_line.png")
        plt.clf()

        print("Saving the model.")
        model.save(f"{directory}/{metric}_{self._modelName}")

        return testResult

    def _calibrateVNF(
        self,
        vnf: str,
        trafficDesignFile: str = "",
        metric: str = "",
        headless: bool = False,
        train: bool = False,
        epochs: int = EPOCHS,
    ) -> None:
        """
        Calibrate the VNF.

        Parameters:
            vnf (str): The VNF to calibrate.
            trafficDesignFile (str): The file containing the design of the traffic generator.
            metric (str): The metric to calibrate.
            train (bool): Specifies if only training should be carried out.
            epochs (int): The number of epochs to train the model for.
            headless (bool): Whether to run the emulator in headless mode.
        """

        epochs = epochs if epochs is not None else EPOCHS
        if not train:

            if not os.path.exists(
                f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
            ):
                os.makedirs(
                    f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
                )
            directory: str = (
                f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
            )
            filename = f"{directory}/calibration_data.csv"

            if trafficDesignFile is not None and trafficDesignFile != "":
                with open(trafficDesignFile, "r", encoding="utf8") as file:
                    trafficDesign: "list[TrafficDesign]" = [json.load(file)]
            else:
                with open(
                    f"{self._config['repoAbsolutePath']}/src/calibrate/traffic-design.json",
                    "r",
                    encoding="utf8",
                ) as file:
                    trafficDesign: "list[TrafficDesign]" = [json.load(file)]

            totalDuration: int = calculateTrafficDuration(trafficDesign[0])

            topology: Topology = {
                "hosts": [
                    {
                        "id": "h1",
                    }
                ],
                "switches": [{"id": "s1"}],
                "links": [
                    {
                        "source": SFCC,
                        "destination": "s1",
                    },
                    {"source": "s1", "destination": "h1"},
                    {"source": "s1", "destination": SERVER},
                ],
            }

            sfcr: SFCRequest = {
                "sfcrID": f"c{vnf.capitalize()}",
                "latency": 10000,
                "vnfs": [vnf],
                "strictOrder": [],
            }

            eg: EmbeddingGraph = {
                "sfcID": f"c{vnf.capitalize()}",
                "vnfs": {
                    "host": {
                        "id": "h1",
                    },
                    "vnf": {"id": vnf},
                    "next": {"host": {"id": SERVER}, "next": TERMINAL},
                },
                "links": [
                    {
                        "source": {"id": SFCC},
                        "destination": {"id": "h1"},
                        "links": ["s1"],
                    },
                    {
                        "source": {"id": "h1"},
                        "destination": {"id": SERVER},
                        "links": ["s1"],
                    },
                ],
            }

            if vnf in self._config["vnfs"]["splitters"]:
                nextVNF: VNF = deepcopy(eg["vnfs"]["next"])
                eg["vnfs"]["next"] = [nextVNF, nextVNF]

            class SFCR(SFCRequestGenerator):
                """
                SFC Request Generator.
                """

                def generateRequests(self) -> None:
                    """
                    Generate the requests.
                    """

                    self._orchestrator.sendRequests([sfcr])

            class SFCSolver(Solver):
                """
                SFC Solver.
                """

                def generateEmbeddingGraphs(self) -> None:
                    """
                    Generate the embedding graphs.
                    """

                    maxRound: int = 1

                    for i in range(maxRound):
                        TUI.appendToSolverLog(f"Round {i + 1}")
                        self._orchestrator.sendEmbeddingGraphs([eg])
                        telemetry: Telemetry = self._orchestrator.getTelemetry()
                        seconds: int = 0
                        hostDataList: "list[HostData]" = []
                        runTime: int = totalDuration + 30  # 30s grace time

                        TUI.appendToSolverLog(f"Waiting for {runTime} seconds.")
                        while seconds < runTime:
                            try:
                                start: float = default_timer()
                                hostData: HostData = telemetry.getHostData()
                                end: float = default_timer()
                                duration: int = round(end - start, 0)
                                hostDataList.append(hostData)
                                seconds += duration
                                TUI.appendToSolverLog(f"{runTime - seconds} seconds left.")
                            except Exception as e:
                                TUI.appendToSolverLog(str(e), True)

                        try:
                            trafficData: pd.DataFrame = self._trafficGenerator.getData(
                                f"{runTime:.0f}s"
                            )
                            hostData: DataFrame = hostDataToFrame(hostDataList)
                            hostData = mergeHostAndTrafficData(hostData, trafficData)
                            if i == 0:
                                hostData.to_csv(filename, index=False)
                            else:
                                hostData.to_csv(filename, mode="a", header=False, index=False)
                        except Exception as e:
                            TUI.appendToSolverLog(str(e), True)
                        self._orchestrator.deleteEmbeddingGraphs([eg])
                        TUI.appendToSolverLog(f"Round {i + 1} finished.")
                    sleep(2)
                    TUI.exit()

            print("Starting OpenRASE.")
            em: SFCEmulator = SFCEmulator(SFCR, SFCSolver, headless)
            print(
                "Running the network and taking measurements. This will take a while."
            )
            em.startTest(topology, trafficDesign)

            print("Training the models.")

            if metric != "":
                self._trainModel(metric, vnf, epochs)
            else:
                self._trainModel(self._headers[0], vnf, epochs)  # cpu
                self._trainModel(self._headers[1], vnf, epochs)  # memory
                self._trainModel(self._headers[2], vnf, epochs)  # latency

            em.end()
        else:
            if metric != "":
                self._trainModel(metric, vnf, epochs)
            else:
                self._trainModel(self._headers[0], vnf, epochs)  # cpu
                self._trainModel(self._headers[1], vnf, epochs)  # memory
                self._trainModel(self._headers[2], vnf, epochs)  # latency

    def _getVNFResourceDemandModel(self, vnf: str, metric: str) -> Any:
        """
        Get the resource demand model of the given metric and VNF.

        Parameters:
            vnf (str): The VNF to get the resource demand model for.
            metric (str): The metric to get the resource demand model for.

        Returns:
            Any: The resource demand model.
        """

        if vnf not in self._models or metric not in self._models[vnf]:
            modelPath: str = f"{self._calDir}/{vnf}/{metric}_{self._modelName}"
            model: Any = tf.keras.models.load_model(modelPath)

            if vnf in self._models:
                self._models[vnf][metric] = model
            else:
                self._models[vnf] = {metric: model}

            return tf.keras.models.load_model(modelPath)
        else:
            return self._models[vnf][metric]

    def _getVNFResourceDemands(self, vnf: str, reqps: float) -> ResourceDemand:
        """
        Get the resource demands of the VNF.

        Parameters:
            vnf (str): The VNF to get the resource demands for.
            reqps (float): The requests per second.

        Returns:
            ResourceDemand: The resource demands.
        """

        cpuModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[0])
        memoryModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[1])
        latencyModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[2])

        cpu: float = cpuModel.predict(np.array([reqps]))[0][0]
        memory: float = memoryModel.predict(np.array([reqps]))[0][0]
        latency: float = latencyModel.predict(np.array([reqps]))[0][0]

        return ResourceDemand(
            cpu=cpu if cpu > 0 else 0, memory=memory if memory > 0 else 0, latency=latency if latency > 0 else 0
        )

    def _getVNFResourceDemandsForRequests(
        self, vnf: str, reqps: "list[float]"
    ) -> ResourceDemand:
        """
        Get the resource demands of the VNF.

        Parameters:
            vnf (str): The VNF to get the resource demands for.
            reqps (list[float]): The requests per second.

        Returns:
            ResourceDemand: The resource demands.
        """

        cpuModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[0])
        memoryModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[1])
        latencyModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[2])

        cpu: float = np.array(cpuModel.predict(np.array(reqps))).flatten()
        memory: float = memoryModel.predict(np.array(reqps)).flatten()
        latency: float = latencyModel.predict(np.array(reqps)).flatten()

        demands: "list[ResourceDemand]" = []
        for cpuPred, memoryPred, latencyPred in zip(cpu, memory, latency):
            cpu = cpuPred
            memory = memoryPred
            latency = latencyPred

            demands.append(
                ResourceDemand(
                    cpu=cpu if cpu > 0 else 0,
                    memory=memory if memory > 0 else 0,
                    latency=latency if latency > 0 else 0,
                )
            )

        return demands

    def calibrateVNFs(
        self,
        trafficDesignFile: str = "",
        vnf: str = "",
        metric: str = "",
        headless: bool = False,
        train: bool = False,
        epochs: int = EPOCHS,
    ) -> None:
        """
        Calibrate all the VNFs.

        Parameters:
            trafficDesignFile (str): The file containing the design of the traffic generator.
            vnf (str): The VNF to calibrate.
            metric (str): The metric to calibrate.
            train (bool): Specifies if only training should be carried out.
            epochs (int): The number of epochs to train the model for.
            headless (bool): Whether to run the emulator in headless mode.
        """

        print("Calibrating VNFs.")
        if vnf != "":
            self._calibrateVNF(vnf, trafficDesignFile, metric, headless, train, epochs)
        else:
            for vnf in self._config["vnfs"]["names"]:
                print("Calibrating VNF: " + vnf)
                self._calibrateVNF(
                    vnf, trafficDesignFile, metric, headless, train, epochs
                )

    def getResourceDemands(self, reqps: float) -> "dict[str, ResourceDemand]":
        """
        Get the resource demands of the VNFs.

        Parameters:
            reqps (float): The requests per second.

        Returns:
            dict[str, ResourceDemand]: The resource demands of each vnf.
        """

        demands: "dict[str, ResourceDemand]" = {}
        for vnf in self._config["vnfs"]["names"]:
            demands[vnf] = self._getVNFResourceDemands(vnf, reqps)

        return demands

    def predictAndCache(self, data: "dict[str, list[float]]", maxDepth=3) -> None:
        """
        Predict on the data and cache the results.

        Parameters:
            data (dict[str, list[float]]): The data to predict on and cache.
            maxDepth (int): The maximum depth of the VNFs.
        """

        uncached: "dict[str, list[float]]" = {}

        for vnf, reqps in data.items():
            requests = []
            requests.extend(reqps)

            for req in reqps:
                for i in range(2, maxDepth + 1):
                    requests.append(int(req / (2 ** (i - 1))))
            requests.append(0)
            if vnf in self._cache:
                for req in requests:
                    intReq: int = int(req)
                    if str(intReq) not in self._cache[vnf]:
                        if vnf in uncached:
                            uncached[vnf].append(intReq)
                        else:
                            uncached[vnf] = [intReq]
            else:
                uncached[vnf] = requests
        for vnf, reqps in uncached.items():
            demands: "list[ResourceDemand]" = self._getVNFResourceDemandsForRequests(
                vnf, reqps
            )
            for req, demand in zip(reqps, demands):
                if vnf in self._cache:
                    self._cache[vnf][str(req)] = demand
                else:
                    self._cache[vnf] = {str(req): demand}

    def getVNFResourceDemandForReqps(self, vnf, reqps: float) -> ResourceDemand:
        """
        Get the resource demands of the VNF for the given requests per second.

        Parameters:
            vnf (str): The VNF to get the resource demands for.
            reqps (float): The requests per second.

        Returns:
            ResourceDemand: The resource demands.
        """

        if vnf not in self._cache or str(int(reqps)) not in self._cache[vnf]:
            demand: ResourceDemand = self._getVNFResourceDemands(vnf, reqps)
            if vnf in self._cache:
                self._cache[vnf][str(int(reqps))] = demand
            else:
                self._cache[vnf] = {str(int(reqps)): demand}
            return demand

        return self._cache[vnf][str(int(reqps))]
