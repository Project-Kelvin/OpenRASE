"""
Defines the functions used to make predictions efficiently.
"""

from typing import Tuple
import numpy as np
from shared.models.embedding_graph import VNF, EmbeddingGraph
from algorithms.hybrid.models.traffic import TimeSFCRequests
from calibrate.demand_predictor import DemandPredictor
from models.calibrate import ResourceDemand
from utils.embedding_graph import traverseVNF

class DemandPredictions:
    """
    Class to handle predictions for resource demands in embedding graphs.
    """

    def __init__(self):
        """
        Initializes the Predictions class.
        """

        self._demandPredictor: DemandPredictor = DemandPredictor()
        self._cpuDemandData: dict[str, float] = {}
        self._memoryDemandData: dict[str, float] = {}

    def _getVNFsInEG(self, eg: "EmbeddingGraph") -> "list[Tuple[str, int]]":
        """
        Get the VNFs in the embedding graph.

        Parameters:
            eg (EmbeddingGraph): The embedding graph to get the VNFs from.
        Returns:
            list[Tuple[str, int]]: A list of tuples containing the VNF name and its depth in the embedding graph.
        """

        vnfList: "list[Tuple[str, int]]" = []
        def parseVNF(vnf: VNF, depth: int) -> None:
            """
            Recursive function to parse VNFs in the embedding graph.

            Parameters:
                vnf (VNF): The VNF to parse.
                depth (int): The depth of the VNF in the embedding graph.
            """

            nonlocal vnfList

            if "vnf" in vnf:
                vnfList.append((vnf["vnf"]["id"], depth))

        traverseVNF(eg["vnfs"], parseVNF)

        return vnfList

    def cacheResourceDemands(
        self,
        egs: "list[EmbeddingGraph]",
        data: TimeSFCRequests
    ) -> None:
        """
        Get the resource demands for each embedding graph based on the provided data.

        Parameters:
            egs (list[EmbeddingGraph]): List of embedding graphs.
            data (TimeSFCRequests): The time series data containing SFC requests.

        Returns:
            None
        """

        cacheData: dict[str, list[float]] = {}
        for eg in egs:
            vnfsList: "list[Tuple[str, int]]" = self._getVNFsInEG(eg)
            reqps = [sfcs[eg["sfcID"]] for sfcs in data]

            for req in reqps:
                for vnf, depth in vnfsList:
                    divisor: float = 2 ** (depth - 1)
                    effectiveReqps: float = req / divisor
                    if vnf not in cacheData:
                        cacheData[vnf] = [effectiveReqps]
                    else:
                        if effectiveReqps not in cacheData[vnf] and f"{vnf}_{str(effectiveReqps)}" not in self._cpuDemandData:
                            cacheData[vnf].append(effectiveReqps)

        if len(cacheData.items()) > 0:
            demandData: tuple[dict[str, float], dict[str, float]] = self._demandPredictor.getVNFResourceDemands(
                cacheData
            )
            self._cpuDemandData.update(demandData[0])
            self._memoryDemandData.update(demandData[1])


    def getDemand(self, vnf: str, reqps: float) -> ResourceDemand:
        """
        Get the resource demand for a specific VNF and requests per second.

        Parameters:
            vnf (str): The VNF to get the resource demand for.
            reqps (float): The requests per second.

        Returns:
            ResourceDemand: The resource demand for the specified VNF and requests per second.
        """

        key: str = f"{vnf}_{str(reqps)}"

        return ResourceDemand(
            cpu=self._cpuDemandData.get(key, 0.0),
            memory=self._memoryDemandData.get(key, 0.0)
        )
