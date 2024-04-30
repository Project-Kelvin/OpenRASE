"""
This code is used to calibrate the CPU, memory, and bandwidth demands of VNFs.
"""

from copy import deepcopy
import os
from time import sleep
from timeit import default_timer
import csv
import json
from typing import Any
from packages.shared.models.embedding_graph import VNF
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
from constants.topology import SERVER, SFCC
from mano.telemetry import Telemetry
from models.calibrate import ResourceDemand
from models.telemetry import HostData
from models.traffic_generator import TrafficData
from sfc.sfc_emulator import SFCEmulator
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.solver import Solver
from utils.traffic_design import calculateTrafficDuration
from utils.tui import TUI


EPOCHS: int = 1000

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
        self._headers: "list[str]" = ["cpu", "memory", "networkIn",
                "networkOut", "ior", "http_reqs", "latency", "duration"]

        if not os.path.exists(f"{self._config['repoAbsolutePath']}/artifacts/calibrations"):
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

        directory: str = f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
        file = f"{directory}/calibration_data.csv"

        directory: str = f"{self._calDir}/{vnf}"

        data: DataFrame = pd.read_csv(file)
        data = data[[metric, "http_reqs"]]

        data = data[(data[metric] != 0) & (data["http_reqs"] != 0)]
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

        reqps: Any = np.array(trainFeatures["http_reqs"])
        normalizer: Any = tf.keras.layers.Normalization(
            input_shape=[1,], axis=None)
        normalizer.adapt(reqps)

        model: Any = tf.keras.Sequential([
            normalizer,
            tf.keras.layers.Dense(units=1)
        ])

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error'
        )

        history: Any = model.fit(
            trainFeatures["http_reqs"],
            trainLabels,
            epochs=epochs,
            verbose=0,
            validation_split=0.2
        )
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel(f'Error [{metric}]')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{directory}/{metric}_loss.png")
        plt.clf()

        testResult: Any = model.evaluate(
            testFeatures["http_reqs"],
            testLabels, verbose=0
        )

        x = tf.linspace(0, 200, 21)
        y = model.predict(x)

        plt.scatter(trainFeatures["http_reqs"], trainLabels, label='Data')
        plt.plot(x, y, color='k', label='Predictions')
        plt.xlabel('reqps')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f"{directory}/{metric}_trend_line.png")
        plt.clf()

        model.save(f"{directory}/{metric}_{self._modelName}")

        return testResult


    def _calibrateVNF(self, vnf: str, trafficDesignFile: str = "", metric: str = "", train: bool = False, epochs: int = EPOCHS) -> None:
        """
        Calibrate the VNF.

        Parameters:
            vnf (str): The VNF to calibrate.
            trafficDesignFile (str): The file containing the design of the traffic generator.
            metric (str): The metric to calibrate.
            train (bool): Specifies if only training should be carried out.
            epochs (int): The number of epochs to train the model for.
        """

        if not train:

            if not os.path.exists(f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"):
                os.makedirs(
                    f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}")
            directory: str = f"{self._config['repoAbsolutePath']}/artifacts/calibrations/{vnf}"
            filename = f"{directory}/calibration_data.csv"

            if trafficDesignFile is not None and trafficDesignFile != "":
                with open(trafficDesignFile, 'r', encoding="utf8") as file:
                    trafficDesign: "list[TrafficDesign]" = [json.load(file)]
            else:
                with open(f"{self._config['repoAbsolutePath']}/src/calibrate/traffic-design.json", 'r', encoding="utf8") as file:
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
                    {
                        "source": "s1",
                        "destination": "h1"
                    },
                    {
                        "source": "s1",
                        "destination": SERVER
                    }
                ]
            }

            sfcr: SFCRequest = {
                "sfcrID": f"c{vnf.capitalize()}",
                "latency": 10000,
                "vnfs": [vnf],
                "strictOrder": []
            }

            eg: EmbeddingGraph = {
                "sfcID": f"c{vnf.capitalize()}",
                "vnfs": {
                    "host": {
                        "id": "h1",
                    },
                    "vnf": {
                        "id": vnf
                    },
                    "next": {
                        "host": {
                            "id": SERVER
                        },
                        "next": TERMINAL
                    }
                },
                "links": [
                    {
                        "source": {
                            "id": SFCC
                        },
                        "destination": {
                            "id": "h1"
                        },
                        "links": ["s1"]
                    },
                    {
                        "source": {
                            "id": "h1"
                        },
                        "destination": {
                            "id": SERVER
                        },
                        "links": ["s1"]
                    }
                ]
            }

            if vnf in self._config["vnfs"]["splitters"]:
                next: VNF = deepcopy(eg["vnfs"]["next"])
                eg["vnfs"]["next"] = [next, next]

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

                _headers: "list[str]" = self._headers

                def generateEmbeddingGraphs(self) -> None:
                    """
                    Generate the embedding graphs.
                    """

                    self._orchestrator.sendEmbeddingGraphs([eg])
                    telemetry: Telemetry = self._orchestrator.getTelemetry()
                    seconds: int = 0

                    # Create a CSV file
                    # Write headers to the CSV file
                    with open(filename, mode='w+', newline='', encoding="utf8") as file:
                        writer = csv.writer(file)
                        writer.writerow(self._headers)
                    while seconds < totalDuration:
                        try:
                            start: float = default_timer()
                            hostData: HostData = telemetry.getHostData()["h1"]["vnfs"]
                            end: float = default_timer()
                            duration: int = round(end - start, 0)
                            trafficData: "dict[str, TrafficData]" = self._trafficGenerator.getData(
                                f"{duration:.0f}s")
                            row = []

                            for _key, data in hostData.items():
                                row.append(data["cpuUsage"][0])
                                row.append(data["memoryUsage"][0]/(1024*1024) if data["memoryUsage"][0] != 0 else 0)
                                row.append(data["networkUsage"][0])
                                row.append(data["networkUsage"][1])

                                if data["networkUsage"][0] == 0 or data["networkUsage"][1] == 0:
                                    row.append(0)
                                else:
                                    row.append((data["networkUsage"][0]/data["networkUsage"][1]))

                                httpReqs: int = trafficData[eg["sfcID"]]["httpReqs"] if eg["sfcID"] in trafficData else 0
                                averageLatency: float = trafficData[eg["sfcID"]]["averageLatency"] if eg["sfcID"] in trafficData else 0
                                httpReqsRate: float = httpReqs / duration if httpReqs != 0 or duration !=0 else 0
                                row.append(httpReqsRate)
                                row.append(averageLatency)

                            TUI.appendToSolverLog(f"{httpReqsRate} requests took {averageLatency} seconds on average.")

                            row.append(f"{round(end - start, 0):.0f}")

                            # Write row data to the CSV file
                            with open(filename, mode='a', newline='', encoding="utf8") as file:
                                writer = csv.writer(file)
                                writer.writerow(row)
                            seconds += duration
                        except Exception as e:
                            TUI.appendToSolverLog(str(e), True)
                    TUI.appendToSolverLog(f"Finished generating traffic for {vnf}.")
                    sleep(2)
                    TUI.exit()

            print("Starting OpenRASE.")
            em: SFCEmulator = SFCEmulator(SFCR, SFCSolver)
            print("Running the network and taking measurements. This will take a while.")
            em.startTest(topology, trafficDesign)

            print("Training the models.")

            if metric != "":
                self._trainModel(metric, vnf, epochs)
            else:
                self._trainModel(self._headers[0], vnf, epochs)  # cpu
                self._trainModel(self._headers[1], vnf, epochs)  # memory
                self._trainModel(self._headers[4], vnf, epochs)  # I/O Ratio

            em.end()
        else:
            if metric != "":
                self._trainModel(metric, vnf, epochs)
            else:
                self._trainModel(self._headers[0], vnf, epochs)  # cpu
                self._trainModel(self._headers[1], vnf, epochs)  # memory
                self._trainModel(self._headers[4], vnf, epochs)  # I/O Ratio


    def _getVNFResourceDemandModel(self, vnf: str, metric: str):
        """
        Get the resource demand model of the given metric and VNF.

        Parameters:
            vnf (str): The VNF to get the resource demand model for.
            metric (str): The metric to get the resource demand model for.
        """

        modelPath: str = f"{self._calDir}/{vnf}/{metric}_{self._modelName}"

        return tf.keras.models.load_model(modelPath)

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
        iorModel: Any = self._getVNFResourceDemandModel(vnf, self._headers[4])

        cpu: float = cpuModel.predict(np.array([reqps]))[0][0]
        memory: float = memoryModel.predict(np.array([reqps]))[0][0]
        ior: float = iorModel.predict(np.array([reqps]))[0][0]

        return ResourceDemand(cpu=cpu, memory=memory, ior=ior)

    def calibrateVNFs(self, trafficDesignFile: str = "", vnf: str = "", metric: str = "", train: bool = False, epochs: int = EPOCHS) -> None:
        """
        Calibrate all the VNFs.

        Parameters:
            trafficDesignFile (str): The file containing the design of the traffic generator.
            vnf (str): The VNF to calibrate.
            metric (str): The metric to calibrate.
            train (bool): Specifies if only training should be carried out.
            epochs (int): The number of epochs to train the model for.
        """
        if vnf != "":
            self._calibrateVNF(vnf, trafficDesignFile, metric, train, epochs)
        else:
            for vnf in self._config["vnfs"]["names"]:
                print("Calibrating VNF: " + vnf)
                self._calibrateVNF(vnf, trafficDesignFile, metric, train, epochs)

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
