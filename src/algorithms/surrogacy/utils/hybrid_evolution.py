"""
This defines the functions used for hybrid evolution.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import os
import random
from time import sleep
import timeit
from typing import Callable, Tuple
import numpy as np
import pandas as pd
import polars as pl
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
import tensorflow as tf
from algorithms.models.embedding import DecodedIndividual, EmbeddingData
from algorithms.surrogacy.models.traffic import TimeSFCRequests
from algorithms.surrogacy.utils.demand_predictions import DemandPredictions
from algorithms.surrogacy.utils.scorer import Scorer
from algorithms.surrogacy.surrogate.surrogate import getSurrogateModel
from models.calibrate import ResourceDemand
from sfc.traffic_generator import TrafficGenerator
from utils.traffic_design import (
    calculateTrafficDuration,
    getTrafficDesignRate,
)
from utils.tui import TUI

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)


class HybridEvolution:
    """
    Class to handle hybrid evolution for resource demand and latency predictions.
    """

    _demandPredictions: DemandPredictions = DemandPredictions()
    _latencyPrediction: np.array = None
    _surrogateModel: tf.keras.Sequential = getSurrogateModel()

    @staticmethod
    def _generateTrafficData(
        egs: "list[EmbeddingGraph]",
        trafficDesign: "list[TrafficDesign]",
        isAvgOnly: bool = False,
        isMaxOnly: bool = False,
    ) -> TimeSFCRequests:
        """
        Generate traffic data for the embedding graphs.

        Parameters:
            egs (list[EmbeddingGraph]): List of embedding graphs.
            trafficDesign (list[TrafficDesign]): Traffic design to use for generating traffic data.
            isAvgOnly (bool): If True, only generate average traffic data.
            isMaxOnly (bool): If True, only generate data for the maximum traffic design.

        Returns:
            TimeSFCRequests: A list of dictionaries containing the traffic data for each embedding graph.
        """

        memoryData: TimeSFCRequests = [
            {
                eg["sfcID"]: float(
                    max(trafficDesign[0], key=lambda x: x["target"])["target"]
                )
                for eg in egs
            }
        ]
        if isMaxOnly:
            return memoryData
        else:
            duration: int = calculateTrafficDuration(trafficDesign[0])
            simulationData: TimeSFCRequests = []
            simTrafficDesign: "list[float]" = getTrafficDesignRate(
                trafficDesign[0], [1] * duration
            )

            if isAvgOnly:
                simulationData = [
                    {eg["sfcID"]: np.mean(simTrafficDesign) for eg in egs}
                ]
            else:
                simulationData = [
                    {eg["sfcID"]: reqps for eg in egs} for reqps in simTrafficDesign
                ]

            return simulationData + memoryData

    @staticmethod
    def _combineEGs(inds: "list[DecodedIndividual]") -> "list[EmbeddingGraph]":
        """
        Combine the embedding graphs from a list of individuals.

        Parameters:
            inds (list[IndividualEG]): List of individuals containing embedding graphs.

        Returns:
            list[EmbeddingGraph]: Combined list of embedding graphs.
        """

        egs: "list[EmbeddingGraph]" = []
        for ind in inds:
            egs.extend(ind[1])

        return egs

    @staticmethod
    def _cacheDemand(
        pop: "list[DecodedIndividual]",
        trafficDesign: "list[TrafficDesign]",
        isMemoryOnly: bool = False,
        isAvgOnly: bool = False,
    ) -> None:
        """
        Cache the demand for score generation.

        Parameters:
            pop (list[IndividualEG]): The population of individuals to cache demands for.
            trafficDesign (list[TrafficDesign]): Traffic design to use for checking memory limits.
            acceptanceRatio (float): The acceptance ratio for the scores.
            isMemoryOnly (bool): If True, only cache memory demands, otherwise cache both memory and traffic demands.
            isAvgOnly (bool): If True, only generate average traffic data.

        Returns:
            None
        """

        egs: "list[EmbeddingGraph]" = copy.deepcopy(HybridEvolution._combineEGs(pop))

        for i, eg in enumerate(egs):
            eg["sfcID"] = f"{eg['sfcID']}_{i}" if "sfcID" in eg else f"sfc{i}"

        memoryData: TimeSFCRequests = HybridEvolution._generateTrafficData(
            egs, trafficDesign, isMaxOnly=True
        )
        if not isMemoryOnly:
            simulationData: TimeSFCRequests = HybridEvolution._generateTrafficData(
                egs, trafficDesign, isMaxOnly=False, isAvgOnly=isAvgOnly
            )

            demandData: TimeSFCRequests = simulationData + memoryData

            HybridEvolution._demandPredictions.cacheResourceDemands(egs, demandData)
        else:
            HybridEvolution._demandPredictions.cacheResourceDemands(egs, memoryData)

    @staticmethod
    def getMaxCpuMemoryUsageOfHosts(
        egs: "list[EmbeddingGraph]",
        topology: Topology,
        embeddingData: EmbeddingData,
        trafficDesign: "list[TrafficDesign]",
    ) -> "tuple[float, float]":
        """
        Get the maximum CPU and memory usage of hosts.

        Parameters:
            egs (list[EmbeddingGraph]): List of embedding graphs to check.
            topology (Topology): The topology to use for checking resource usage.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): Embedding data containing VNF and depth information.
            trafficDesign (list[TrafficDesign]): Traffic design to use for checking resource usage.

        Returns:
            tuple[float, float]: Maximum CPU and memory usage of hosts.
        """

        data: pl.DataFrame = HybridEvolution._generateTrafficData(
            egs, trafficDesign, isMaxOnly=True
        )

        scores: "dict[str, ResourceDemand]" = Scorer.getHostScores(
            data, topology, embeddingData, HybridEvolution._demandPredictions
        )
        maxMemory: float = max(
            [score["memory"] for score in scores.values()], default=0
        )
        maxCPU: float = max([score["cpu"] for score in scores.values()], default=0)

        return maxCPU, maxMemory

    @staticmethod
    def doesExceedMemoryLimit(
        egs: "list[EmbeddingGraph]",
        topology: Topology,
        embeddingData: EmbeddingData,
        trafficDesign: "list[TrafficDesign]",
        maxMemoryDemand: int,
    ) -> bool:
        """
        Check if the memory limit is exceeded.

        Parameters:
            egs (list[EmbeddingGraph]): List of embedding graphs to check.
            topology (Topology): The topology to use for checking memory limits.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): Embedding data containing VNF and depth information.
            trafficDesign (list[TrafficDesign]): Traffic design to use for checking memory limits.
            maxMemoryDemand (int): The maximum memory demand allowed.

        Returns:
            bool: True if the memory limit is exceeded, False otherwise.
        """

        _maxCPU, maxMemory = HybridEvolution.getMaxCpuMemoryUsageOfHosts(
            egs, topology, embeddingData, trafficDesign
        )

        return maxMemory > maxMemoryDemand

    @staticmethod
    def _generateScores(
        trafficDesign: "list[TrafficDesign]",
        egs: "list[EmbeddingGraph]",
        topology: Topology,
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        linkData: "dict[str, dict[str, float]]",
        isAvgOnly: bool = False,
    ) -> np.array:
        """
        Generate scores for the given traffic design and embedding graphs.

        Parameters:
            trafficDesign (list[TrafficDesign]): The traffic design to use for generating scores.
            egs (list[EmbeddingGraph]): List of embedding graphs to use for generating scores.
            gen (int): The generation number.
            individualIndex (int): The index of the individual.
            topology (Topology): The topology to use for generating scores.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): Embedding data containing VNF and depth information.
            linkData (dict[str, dict[str, float]]): Link data containing link capacities.
            isAvgOnly (bool): If True, only generate average scores.

        Returns:
            np.array: A numpy array containing the generated scores.
        """

        simulationData: pl.DataFrame = HybridEvolution._generateTrafficData(
            egs, trafficDesign, isMaxOnly=False, isAvgOnly=isAvgOnly
        )

        scores: np.array = Scorer.getSFCScores(
            simulationData,
            topology,
            egs,
            embeddingData,
            linkData,
            HybridEvolution._demandPredictions,
        )

        return scores

    @staticmethod
    def _generateMeanScores(
        trafficDesign: "list[TrafficDesign]",
        egs: "list[EmbeddingGraph]",
        gen: int,
        individualIndex: int,
        topology: Topology,
        embeddingData: "dict[str, dict[str, list[Tuple[str, int]]]]",
        linkData: "dict[str, dict[str, float]]",
    ) -> np.array:
        """
        Generate scores for the given traffic design and embedding graphs.

        Parameters:
            trafficDesign (list[TrafficDesign]): The traffic design to use for generating scores.
            egs (list[EmbeddingGraph]): List of embedding graphs to use for generating scores.
            gen (int): The generation number.
            individualIndex (int): The index of the individual.
            topology (Topology): The topology to use for generating scores.
            embeddingData (dict[str, dict[str, list[Tuple[str, int]]]]): Embedding data containing VNF and depth information.
            linkData (dict[str, dict[str, float]]): Link data containing link capacities.
            acceptanceRatio (float): The acceptance ratio for the scores.

        Returns:
            np.array: A numpy array containing the generation index, individual index, mean link score and CPU score.
        """

        try:
            scores: np.array = HybridEvolution._generateScores(
                trafficDesign,
                egs,
                topology,
                embeddingData,
                linkData,
                isAvgOnly=True,
            )
        except Exception as e:
            TUI.appendToSolverLog(f"Error generating scores: {e}")
            return pl.DataFrame()

        mean_link = np.mean(scores[:]["link"])
        mean_cpu = np.mean(scores[:]["max_cpu"])

        dt = np.dtype(
            [
                ("generation", np.int32),
                ("individual", np.int32),
                ("mean_link", np.float64),
                ("mean_cpu", np.float64),
                ("latency", np.float64),
            ]
        )
        return np.array([(gen, individualIndex, mean_link, mean_cpu, 0.0)], dtype=dt)

    @staticmethod
    def _predictLatency(
        inds: "list[DecodedIndividual]",
        trafficDesign: "list[TrafficDesign]",
        gen: int,
        topology: Topology,
    ) -> None:
        """
        Predict the latency for the given individuals.

        Parameters:
            inds (list[IndividualEG]): List of individuals to predict latency for.
            trafficDesign (list[TrafficDesign]): The traffic design to use for predicting latency.
            gen (int): The generation number.
            topology (Topology): The topology to use for predicting latency.

        Returns:
            None
        """

        start: float = timeit.default_timer()
        scores: np.array = None

        with ProcessPoolExecutor() as executor:
            try:
                futures = [
                    executor.submit(
                        HybridEvolution._generateMeanScores,
                        trafficDesign,
                        ind[1],
                        gen,
                        ind[0],
                        topology,
                        ind[2],
                        ind[3],
                    )
                    for ind in inds
                    if len(ind[1]) > 0
                ]
            except Exception as e:
                TUI.appendToSolverLog(f"Error generating mean scores: {e}")

            for future in as_completed(futures):
                if scores is None:
                    scores = future.result()
                else:
                    scores = np.concatenate((scores, future.result()))

        inputData: np.array = np.concatenate(
            (
                scores[:]["mean_link"].reshape(-1, 1),
                scores[:]["mean_cpu"].reshape(-1, 1),
            ),
            axis=1,
        )

        data: np.array = HybridEvolution._surrogateModel.predict(inputData, verbose=0)

        scores[:]["latency"] = data[:, 0]

        if HybridEvolution._latencyPrediction is None:
            HybridEvolution._latencyPrediction = scores
        else:
            HybridEvolution._latencyPrediction = np.concatenate(
                (HybridEvolution._latencyPrediction, scores)
            )

        end: float = timeit.default_timer()
        TUI.appendToSolverLog(
            f"Latency prediction took {end - start:.2f} seconds for generation {gen}."
        )

    @staticmethod
    def getPredictedLatency(gen: int, ind: int) -> float:
        """
        Get the predicted latency.

        Parameters:
            gen (int): The generation number.
            ind (int): The index of the individual.

        Returns:
            float: The predicted latency for the specified generation and individual.
        """

        if HybridEvolution._latencyPrediction is None:
            return None

        mask = (HybridEvolution._latencyPrediction[:]["generation"] == gen) & (
            HybridEvolution._latencyPrediction[:]["individual"] == ind
        )
        rows = HybridEvolution._latencyPrediction[mask]

        return rows[0]["latency"] if rows.shape[0] > 0 else 0.0

    @staticmethod
    def cacheForOffline(
        pop: "list[DecodedIndividual]",
        trafficDesign: "list[TrafficDesign]",
        topology: Topology,
        gen: int,
        isAvgOnly: bool = False
    ) -> None:
        """
        Evaluate the offline performance of the population.

        Parameters:
            pop (list[IndividualEG]): The population to evaluate.
            trafficDesign (list[TrafficDesign]): The traffic design to use for evaluation.
            topology (Topology): The topology to use for evaluation.
            gen (int): The generation number.
            isAvgOnly (bool): If True, only generate average traffic data.

        Returns:
            None
        """

        HybridEvolution._cacheDemand(pop, trafficDesign, isAvgOnly=isAvgOnly)
        HybridEvolution._predictLatency(pop, trafficDesign, gen, topology)

    @staticmethod
    def cacheForOnline(
        pop: "list[DecodedIndividual]", trafficDesign: "list[TrafficDesign]"
    ) -> None:
        """
        Evaluate the online performance of the population.

        Parameters:
            pop (list[IndividualEG]): The population to evaluate.
            trafficDesign (list[TrafficDesign]): The traffic design to use for evaluation.

        Returns:
            None
        """

        HybridEvolution._cacheDemand(pop, trafficDesign, isMemoryOnly=True)

    @staticmethod
    def evaluationOnSurrogate(
        individual: DecodedIndividual,
        gen: int,
        ngen: int,
        topology: Topology,
        trafficDesign: "list[TrafficDesign]",
        maxMemoryDemand: int,
    ) -> "tuple[int, float, float]":
        """
        Evaluate the individual using the BeNNS.

        Parameters:
            individual (IndividualEG): The individual to evaluate.
            gen (int): The current generation number.
            ngen (int): The total number of generations.
            topology (Topology): The topology to use for evaluation.
            trafficDesign (list[TrafficDesign]): The traffic design to use for evaluation.
            maxMemoryDemand (int): The maximum memory demand allowed.

        Returns:
            tuple[int, float, float]: the fitness values (individual index, acceptance ratio and latency).
        """

        penaltyLatency: int = 50000
        egs: "list[EmbeddingGraph]" = individual[1]
        acceptanceRatio: float = individual[4]

        latency: float = 0

        if len(egs) > 0:
            if HybridEvolution.doesExceedMemoryLimit(
                individual[1], topology, individual[2], trafficDesign, maxMemoryDemand
            ):
                penalty: float = gen / ngen
                acceptanceRatio = (
                    acceptanceRatio - penalty if len(egs) > 0 else acceptanceRatio
                )
                latency = penaltyLatency * penalty if len(egs) > 0 else penaltyLatency

                return (individual[0], acceptanceRatio, latency)

            latency: float = HybridEvolution.getPredictedLatency(gen, individual[0])
        else:
            penalty: float = gen / ngen
            acceptanceRatio = (
                acceptanceRatio - penalty if len(egs) > 0 else acceptanceRatio
            )
            latency = penaltyLatency * penalty if len(egs) > 0 else penaltyLatency

        return (individual[0], acceptanceRatio, round(latency))

    @staticmethod
    def evaluationOnEmulator(
        individual: DecodedIndividual,
        fgrs: "list[EmbeddingGraph]",
        gen: int,
        ngen: int,
        sendEGs: "Callable[[list[EmbeddingGraph]], None]",
        deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
        trafficDesign: "list[TrafficDesign]",
        trafficGenerator: TrafficGenerator,
        topology: Topology,
        maxMemoryDemand: int,
    ) -> "tuple[float, float]":
        """
        Evaluate the individual.

        Parameters:
            individual (list[list[int]]): the individual to evaluate.
            fgrs (list[EmbeddingGraph]): The SFC Requests.
            gen (int): the generation.
            ngen (int): the number of generations.
            sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
            deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
            trafficDesign (list[TrafficDesign]): The Traffic Design.
            trafficGenerator (TrafficGenerator): The Traffic Generator.
            topology (Topology): The Topology.
            maxMemoryDemand (int): The maximum memory demand.

        Returns:
            tuple[float, float]: the fitness values (acceptance ratio and latency).
        """

        acceptanceRatio: float = individual[4]
        TUI.appendToSolverLog(
            f"Acceptance Ratio: {len(individual[1])}/{len(fgrs)} = {acceptanceRatio}"
        )

        penaltyLatency: int = 50000

        if len(individual[1]) > 0:
            if HybridEvolution.doesExceedMemoryLimit(
                individual[1], topology, individual[2], trafficDesign, maxMemoryDemand
            ):
                penalty: float = gen / ngen
                acceptanceRatio = (
                    acceptanceRatio - penalty
                    if len(individual[1]) > 0
                    else acceptanceRatio
                )
                latency = (
                    penaltyLatency * penalty
                    if len(individual[1]) > 0
                    else penaltyLatency
                )

                return (acceptanceRatio, latency)

            sendEGs(individual[1])

            duration: int = calculateTrafficDuration(trafficDesign[0])
            TUI.appendToSolverLog(f"Traffic Duration: {duration}s")
            TUI.appendToSolverLog(f"Waiting for {duration}s...")
            sleep(duration)
            TUI.appendToSolverLog(f"Done waiting for {duration}s.")

            trafficData: pd.DataFrame = trafficGenerator.getData(f"{duration:.0f}s")

            latency: float = 0

            if (
                trafficData.empty
                or "_time" not in trafficData.columns
                or "_value" not in trafficData.columns
            ):
                TUI.appendToSolverLog("Traffic data is empty.")

                latency = penaltyLatency
            else:
                trafficData["_time"] = trafficData["_time"] // 1000000000

                groupedTrafficData: pd.DataFrame = trafficData.groupby(
                    ["_time", "sfcID"]
                ).agg(
                    reqps=("_value", "count"),
                    medianLatency=("_value", "median"),
                )

                latency: float = groupedTrafficData["medianLatency"].mean()

            TUI.appendToSolverLog(f"Deleting graphs belonging to generation {gen}")
            deleteEGs(individual[1])
        else:
            penalty: float = gen / ngen
            acceptanceRatio = (
                acceptanceRatio - penalty if len(individual[1]) > 0 else acceptanceRatio
            )
            latency = (
                penaltyLatency * penalty if len(individual[1]) > 0 else penaltyLatency
            )

        TUI.appendToSolverLog(f"Latency: {latency}ms")

        return (acceptanceRatio, round(latency))

    @staticmethod
    def generateScoresForRealTrafficData(
        ind: DecodedIndividual,
        trafficData: pd.DataFrame,
        trafficDesign: "list[TrafficDesign]",
        topology: Topology,
    ) -> pl.DataFrame:
        """
        Generate scores for the real traffic data.

        Parameters:
            ind (DecodedIndividual): The decoded individual to evaluate.
            trafficData (pd.DataFrame): The traffic data to generate scores for.
            trafficDesign (list[TrafficDesign]): The traffic design to use for generating scores.
            gen (int): The generation number.
            topology (Topology): The topology to use for generating scores.

        Returns:
            pl.DataFrame: A DataFrame containing the generated scores.
        """

        if trafficData.empty:
            return pl.DataFrame()

        trafficData["_time"] = trafficData["_time"] // 1000000000

        groupedTrafficData: pd.DataFrame = trafficData.groupby(["_time", "sfcID"]).agg(
            reqps=("_value", "count"),
            medianLatency=("_value", "median"),
        )

        scores: np.array = HybridEvolution._generateScores(
            trafficDesign, ind[1], topology, ind[2], ind[3]
        )

        scores = pl.DataFrame(scores)

        scores = scores.with_columns(
            pl.lit(1).alias("ar"),
            pl.lit(0.0).alias("real_reqps"),
            pl.lit(0.0).alias("latency"),
        )

        index: int = 0
        for i, group in groupedTrafficData.groupby(level=0):
            for eg in ind[1]:
                scores = scores.with_columns(
                    pl.when((pl.col("sfc") == eg["sfcID"]) & (pl.col("time") == index))
                    .then(
                        group.loc[(i, eg["sfcID"])]["reqps"]
                        if eg["sfcID"] in group.index.get_level_values(1)
                        else 0
                    )
                    .otherwise(pl.col("real_reqps"))
                    .alias("real_reqps"),
                    pl.when((pl.col("sfc") == eg["sfcID"]) & (pl.col("time") == index))
                    .then(
                        group.loc[(i, eg["sfcID"])]["medianLatency"]
                        if eg["sfcID"] in group.index.get_level_values(1)
                        else 0
                    )
                    .otherwise(pl.col("latency"))
                    .alias("latency"),
                )
            index += 1

        return scores
