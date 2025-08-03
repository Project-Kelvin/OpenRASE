"""
Defines the demand prediction class.
"""

from typing import Any
import numpy as np
import polars as pl
from shared.models.config import Config
from shared.utils.config import getConfig
import tensorflow as tf
from calibrate.constants import CALIBRATION_DIR, METRICS, MODEL_NAME
from models.calibrate import ResourceDemand


class DemandPredictor:
    """
    A class to predict resource demand based on traffic data.
    """

    def __init__(self):
        self._models: "dict[str, dict[str, Any]]" = {}
        config: Config = getConfig()
        vnfs: "list[str]" = config["vnfs"]["names"]

        for vnf in vnfs:
            for metric in METRICS:
                self._getVNFResourceDemandModel(vnf, metric)

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
            modelPath: str = f"{CALIBRATION_DIR}/{vnf}/{metric}_{MODEL_NAME}"
            model: Any = tf.keras.models.load_model(modelPath)

            if vnf in self._models:
                self._models[vnf][metric] = model
            else:
                self._models[vnf] = {metric: model}

            return tf.keras.models.load_model(modelPath)
        else:
            return self._models[vnf][metric]

    def getVNFResourceDemands(self, data: dict[str, list[float]]) -> "tuple[dict[str, float], dict[str, float]]":
        """
        Get the resource demands of the VNF.

        Parameters:
            data (dict[str, list[float]]): The data to get the resource demands for.

        Returns:
            tuple[dict[str, float], dict[str, float]]: Dataframe with the resource demands of the VNF.
        """

        cpuData: dict[str, float] = {}
        memoryData: dict[str, float] = {}

        for vnf, reqps in data.items():
            cpus: "list[float]" = self._models[vnf][METRICS[0]].predict(
                np.array(reqps), verbose=0
            ).reshape(1, -1)[0].tolist()
            memories: "list[float]" = self._models[vnf][METRICS[1]].predict(
                np.array(reqps), verbose=0
            ).reshape(1, -1)[0].tolist()

            for req, cpu, memory in zip(reqps, cpus, memories):
                key = f"{vnf}_{str(req)}"
                if key not in cpuData:
                    cpuData[key] = cpu

                if key not in memoryData:
                    memoryData[key] = memory

        return (cpuData, memoryData)

    def getResourceDemands(self, reqps: int) -> "dict[str, ResourceDemand]":
        """
        Get the resource demands of all VNFs for a specific requests per second.

        Parameters:
            reqps (int): The requests per second to get the resource demands for.

        Returns:
            dict[str, ResourceDemand]: A dictionary with the resource demands of all VNFs.
        """

        resourceDemands: "dict[str, ResourceDemand]" = {}
        config: Config = getConfig()
        vnfs: "list[str]" = config["vnfs"]["names"]
        for vnf in vnfs:
            cpu: float = self._models[vnf][METRICS[0]].predict(
                np.array([reqps]), verbose=0
            ).reshape(1, -1)[0][0]
            memory: float = self._models[vnf][METRICS[1]].predict(
                np.array([reqps]), verbose=0
            ).reshape(1, -1)[0][0]

            resourceDemands[vnf] = ResourceDemand(cpu=cpu, memory=memory)

        return resourceDemands

    def getResourceDemandsOfVNF(self, vnf: str, reqps: int) -> ResourceDemand:
        """
        Get the resource demands of a specific VNF for a specific requests per second.

        Parameters:
            vnf (str): The VNF to get the resource demands for.
            reqps (int): The requests per second to get the resource demands for.

        Returns:
            ResourceDemand: The resource demand of the specified VNF.
        """

        cpu: float = self._models[vnf][METRICS[0]].predict(
            np.array([reqps]), verbose=0
        ).reshape(1, -1)[0][0]
        memory: float = self._models[vnf][METRICS[1]].predict(
            np.array([reqps]), verbose=0
        ).reshape(1, -1)[0][0]

        return ResourceDemand(cpu=cpu, memory=memory)
