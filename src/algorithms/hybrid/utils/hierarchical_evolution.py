"""
Defines the class responsible for the hierarchical evolution.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import os
import random
import timeit
from typing import Callable, cast
from uuid import uuid4
from deap import base, tools
import numpy as np
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig

from algorithms.hybrid.constants.genesis_objective import LATENCY, POWER
from algorithms.hybrid.models.individuals import GenesisIndividual, Individual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from algorithms.models.embedding import DecodedIndividual
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI


class HierarchicalEvolution:
    """
    Class responsible for the hierarchical evolution.
    """

    _metaPopulation: list[Individual] = []
    _genesisPopulation: list[GenesisIndividual] = []

    def __init__(
        self,
        metaPopSize: int,
        genesisPopSize: int,
        metaCxPb: float,
        genesisCxPb: float,
        metaMutPb: float,
        genesisMutPb: float,
        metaIndPb: float,
        genesisIndPb: float,
        metaMaxGen: int,
        genesisMaxGen: int,
        minAR: float,
        maxSecondObjective: int,
        noOfNeurons: int,
        maxMemoryDemand: int,
        minQualInd: int,
        sfcrs: list[SFCRequest],
        topology: Topology,
        trafficDesign: list[TrafficDesign],
        trafficGenerator: TrafficGenerator,
        telemetry: Telemetry,
        objectiveType: str,
        dirName: str,
        experimentName: str,
        sendEGs: Callable[[list[EmbeddingGraph]], None],
        deleteEGs: Callable[[list[EmbeddingGraph]], None],
        retainPopulation: bool = False,
    ) -> None:
        """
        Initializes the hierarchical evolution.

        Parameters:
            metaPopSize (int): the size of the meta-population.
            genesisPopSize (int): the size of the genesis population for each meta-individual.
            metaCxPb (float): the probability of mating two meta-individuals.
            metaMutPb (float): the probability of mutating a meta-individual.
            metaIndPb (float): the independent probability for each attribute to be mutated in a meta-individual.
            genesisIndPb (float): the independent probability for each attribute to be mutated in a genesis individual.
            metaMaxGen (int): the maximum number of generations for the meta-evolution.
            genesisMaxGen (int): the maximum number of generations for the genesis evolution.
            minAR (float): the minimum AR to consider an individual qualified.
            maxSecondObjective (int): the maximum value for the second objective to consider an individual qualified.
            noOfNeurons (int): the number of neurons in the hidden layer of the Neural Network.
            maxMemoryDemand (int): the maximum memory demand for an embedding to be considered feasible.
            minQualInd (int): the minimum number of qualified individuals to finish the evolution.
            sfcrs (list[SFCRequest]): the list of Service Function Chains.
            topology (Topology): the topology.
            trafficDesign (list[TrafficDesign]): the traffic design.
            trafficGenerator (TrafficGenerator): the traffic generator.
            telemetry (Telemetry): telemetry instance.
            objectiveType (str): the type of objective to optimize, either latency or power consumption.
            dirName (str): the directory name for logging.
            experimentName (str): the name of the experiment for logging.
            sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
            deleteEGs (Callable[[list[EmbeddingGraph]], None]): the function to delete the Embedding Graphs.
            retainPopulation (bool): specifies if the population should be retained in memory after evolution.

        Returns:
            None
        """

        self._toolbox: base.Toolbox = base.Toolbox()
        self._pfs: str = ""
        self._fitness: str = ""
        self._meta: str = ""
        self._experimentDir: str = ""
        self._metaPopSize: int = metaPopSize
        self._genesisPopSize: int = genesisPopSize
        self._metaCxPb: float = metaCxPb
        self._metaMutPb: float = metaMutPb
        self._metaIndPb: float = metaIndPb
        self._genesisIndPb: float = genesisIndPb
        self._metaMaxGen: int = metaMaxGen
        self._genesisMaxGen: int = genesisMaxGen
        self._minAR: float = minAR
        self._maxSecondObjective: int = maxSecondObjective
        self._noOfNeurons: int = noOfNeurons
        self._maxMemoryDemand: int = maxMemoryDemand
        self._sfcrs: list[SFCRequest] = sfcrs
        self._topology: Topology = topology
        self._trafficDesign: list[TrafficDesign] = trafficDesign
        self._trafficGenerator: TrafficGenerator = trafficGenerator
        self._telemetry: Telemetry = telemetry
        self._objectiveType: str = objectiveType
        self._dirName: str = dirName
        self._experimentName: str = experimentName
        self._retainPopulation: bool = retainPopulation
        self._minQualInd: int = minQualInd
        self._genesisMutPb: float = genesisMutPb
        self._genesisCxPb: float = genesisCxPb
        self._sendEGs: Callable[[list[EmbeddingGraph]], None] = sendEGs
        self._deleteEGs: Callable[[list[EmbeddingGraph]], None] = deleteEGs

    def _createLogFiles(
        self, dirName: str, experimentName: str, secondObjective: str
    ) -> None:
        """
        Creates the log files for the hierarchical evolution.

        Parameters:
            dirName (str): the directory name.
            experimentName (str): the name of the experiment.
            secondObjective (str): the name of the second objective.

        Returns:
            None
        """

        config: Config = getConfig()
        artifactsDir: str = os.path.join(config["repoAbsolutePath"], "artifacts")
        if not os.path.exists(artifactsDir):
            os.mkdir(artifactsDir)
        experimentsDir: str = os.path.join(artifactsDir, "experiments")
        if not os.path.exists(experimentsDir):
            os.mkdir(experimentsDir)
        algorithmDir: str = os.path.join(experimentsDir, dirName)
        if not os.path.exists(algorithmDir):
            os.mkdir(algorithmDir)
        experimentDir: str = os.path.join(algorithmDir, experimentName)
        if not os.path.exists(experimentDir):
            os.mkdir(experimentDir)

        self._experimentDir = experimentDir

        self._pfs = os.path.join(experimentDir, "pfs.csv")
        self._fitness = os.path.join(experimentDir, "fitness.csv")
        self._meta = os.path.join(experimentDir, "meta.csv")

        with open(self._pfs, "w") as pfFile:
            pfFile.write(
                f"method,meta_generation,genesis_generation,ar,{secondObjective}\n"
            )
        with open(self._fitness, "w") as fitnessFile:
            fitnessFile.write(
                f"method,meta_generation,genesis_generation,average_ar,max_ar,min_ar,average_{secondObjective},max_{secondObjective},min_{secondObjective}\n"
            )
        with open(self._meta, "w") as metaFile:
            metaFile.write(
                "meta_generation,average_rejection_rate,max_rejection_rate,min_rejection_rate,average_sigma,max_sigma,min_sigma\n"
            )

    def _writeFitnessLog(
        self,
        metaGen: int,
        genesisGen: int,
        ars: list[float],
        secondObjectives: list[float],
        method: str,
    ) -> None:
        """
        Writes the fitness log for the hierarchical evolution.

        Parameters:
            metaGen (int): the current meta-generation.
            genesisGen (int): the current genesis generation.
            ars (list[float]): the list of ARs of the population.
            secondObjectives (list[float]): the list of second objective values of the population.
            method (str): the method name.

        Returns:
            None
        """

        with open(self._fitness, "a") as fitnessFile:
            fitnessFile.write(
                f"{method},{metaGen},{genesisGen},{np.mean(ars)},{max(ars)},{min(ars)},{np.mean(secondObjectives)},{max(secondObjectives)},{min(secondObjectives)}\n"
            )

    def _writePFLog(
        self, metaGen: int, genesisGen: int, hof: tools.ParetoFront, method: str
    ) -> None:
        """
        Writes the Pareto front log for the hierarchical evolution.

        Parameters:
            metaGen (int): the current meta-generation.
            genesisGen (int): the current genesis generation.
            hof (tools.ParetoFront): the Pareto front of the population.
            method (str): the method name.

        Returns:
            None
        """

        with open(self._pfs, "a") as pfFile:
            for individual in hof:
                pfFile.write(
                    f"{method},{metaGen},{genesisGen},{individual.fitness.values[0]},{individual.fitness.values[1]}\n"
                )

    def _writeMetaLog(self, metaGen: int, metaPopulation: list[Individual]) -> None:
        """
        Writes the meta log for the hierarchical evolution.

        Parameters:
            metaGen (int): the current meta-generation.
            metaPopulation (list[Individual]): the meta-population.

        Returns:
            None
        """

        with open(self._meta, "a") as metaFile:
            rejectionRates: list[float] = [ind[0] for ind in metaPopulation]
            sigmas: list[float] = [ind[1] for ind in metaPopulation]
            metaFile.write(
                f"{metaGen},{np.mean(rejectionRates)},{max(rejectionRates)},{min(rejectionRates)},{np.mean(sigmas)},{max(sigmas)},{min(sigmas)}\n"
            )

    def _generateHyperParameters(self) -> list[float]:
        """
        Generates random hyper-parameters.

        Returns:
            list[float]: the generated hyper-parameters.
        """

        return [round(random.uniform(0, 1), 2), round(random.uniform(0, 10), 2)]

    def _generateRandomMetaIndividual(self) -> Individual:
        """
        Generates a random meta-individual.

        Returns:
            Individual: a random meta-individual.
        """

        return Individual(self._generateHyperParameters())

    def _metaMutate(self, individual: Individual, indpb: float) -> tuple[Individual]:
        """
        Mutates a meta-individual.

        Parameters:
            individual (Individual): the meta-individual to mutate.
            indpb (float): the independent probability for each attribute to be mutated.

        Returns:
            tuple[Individual]: the mutated meta-individual.
        """

        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = self._generateHyperParameters()[i]

        return (individual,)

    def _generateGenesisPopulation(
        self, metaIndividual: Individual
    ) -> list[GenesisIndividual]:
        """
        Generates the genesis population.

        Parameters:
            metaIndividual (Individual): the meta-individual to use for generating the genesis population.

        Returns:
            list[GenesisIndividual]: the generated genesis population.
        """

        population: list[GenesisIndividual] = []
        for _ in range(self._genesisPopSize):
            individual: GenesisIndividual = cast(
                GenesisIndividual,
                GenesisUtils.generateRandomGenesisIndividual(
                    GenesisIndividual, self._topology, self._sfcrs
                ),
            )
            individual.metaIndividual = metaIndividual
            population.append(individual)

        return population

    def _initialiseMetaEvolver(self) -> None:
        """
        Initializes the hyper-evolver.

        Returns:
            None
        """

        self._toolbox.register("metaIndividual", self._generateRandomMetaIndividual)
        self._toolbox.register(
            "metaPopulation", tools.initRepeat, list, self._toolbox.metaIndividual
        )
        self._toolbox.register("metaMate", tools.cxBlend, alpha=0.5)
        self._toolbox.register("metaMutate", self._metaMutate, indpb=self._metaIndPb)
        self._toolbox.register("select", tools.selNSGA2)
        self._toolbox.register("genesisMate", GenesisUtils.genesisCrossover)
        self._toolbox.register(
            "genesisMutate", GenesisUtils.genesisMutate, indpb=self._genesisIndPb
        )

    def _generateMetaOffspring(
        self, population: list[Individual], cxpb: float, mutpb: float
    ) -> list[Individual]:
        """
        Generates the meta-offspring.

        Parameters:
            population (list[Individual]): the meta-population to generate the offspring from.
            cxpb (float): the probability of mating two individuals.
            mutpb (float): the probability of mutating an individual.

        Returns:
            list[Individual]: the generated meta-offspring.
        """

        offspring: list[Individual] = deepcopy(population)

        for child in offspring:
            if random.random() < cxpb:
                child2: Individual = random.choice(
                    [children for children in offspring if children.id != child.id]
                )
                self._toolbox.metaMate(child, child2)

                del child.fitness.values
                del child2.fitness.values
                child.id = uuid4()
                child2.id = uuid4()

        for mutant in offspring:
            if random.random() < mutpb:
                self._toolbox.metaMutate(mutant)

                del mutant.fitness.values

        return offspring

    def _updateMetaIndividualOfGenesisPopulation(
        self,
        metaPopulation: list[Individual],
        genesisPopulation: list[GenesisIndividual],
    ) -> list[GenesisIndividual]:
        """
        Updates the meta-individual of the genesis population.

        Parameters:
            metaPopulation (list[Individual]): the meta-population.
            genesisPopulation (list[GenesisIndividual]): the genesis population to update.

        Returns:
            list[GenesisIndividual]: the updated genesis population.
        """

        metaPopSize: int = len(metaPopulation)
        genesisPopSize: int = len(genesisPopulation)
        genesisPopPerMetaInd: int = genesisPopSize // metaPopSize
        genesisPopMetaBoundary: int = genesisPopPerMetaInd

        for metaIndex, _ in enumerate(metaPopulation):
            for genesisInd in genesisPopulation[
                metaIndex * genesisPopPerMetaInd : genesisPopMetaBoundary
            ]:
                genesisInd.metaIndividual = metaPopulation[metaIndex]

            genesisPopMetaBoundary += genesisPopPerMetaInd

        return genesisPopulation

    def _evaluateMetaFitness(
        self,
        metaPopulation: list[Individual],
        genesisPopulation: list[GenesisIndividual],
    ) -> list[Individual]:
        """
        Evaluates the fitness of the meta-individuals.

        Parameters:
            metaPopulation (list[Individual]): the meta-population.
            genesisPopulation (list[GenesisIndividual]): the genesis population.

        Returns:
            list[Individual]: the meta-population with evaluated fitness.
        """

        metaPopSize: int = len(metaPopulation)
        genesisPopSize: int = len(genesisPopulation)
        genesisPopPerMetaInd: int = genesisPopSize // metaPopSize
        genesisPopMetaBoundary: int = genesisPopPerMetaInd

        for metaIndex, _metaIndividual in enumerate(metaPopulation):
            medianAR: float = cast(
                float,
                np.median(
                    [
                        cast(float, genesisInd.fitness.values[0])
                        for genesisInd in genesisPopulation[
                            metaIndex * genesisPopPerMetaInd : genesisPopPerMetaInd
                        ]
                    ]
                ),
            )
            medianLatency: float = cast(
                float,
                np.median(
                    [
                        cast(float, genesisInd.fitness.values[1])
                        for genesisInd in genesisPopulation[
                            metaIndex * genesisPopPerMetaInd : genesisPopMetaBoundary
                        ]
                    ]
                ),
            )

            metaPopulation[metaIndex].fitness.values = (medianAR, medianLatency)
            genesisPopMetaBoundary += genesisPopPerMetaInd

        return metaPopulation

    def _selectMetaPopulation(
        self, parents: list[Individual], offspring: list[Individual]
    ) -> list[Individual]:
        """
        Selects the meta-population for the next generation.

        Parameters:
            parents (list[Individual]): the parent meta-individuals.
            offspring (list[Individual]): the offspring meta-individuals.

        Returns:
            list[Individual]: the selected meta-population for the next generation.
        """

        newPop: list[Individual] = []
        for parent, child in zip(parents, offspring):
            newPop.append(self._toolbox.select([parent, child], k=1)[0])

        return newPop

    def _generateGenesisOffspring(
        self,
        genesisPopulation: list[GenesisIndividual],
        metaPopulation: list[Individual],
    ) -> list[GenesisIndividual]:
        """
        Generates the genesis offspring.

        Parameters:
            genesisPopulation (list[GenesisIndividual]): the genesis population to generate the offspring from.
            metaPopulation (list[Individual]): the meta-population.
            cxpb (float): the probability of mating two individuals.
            mutpb (float): the probability of mutating an individual.

        Returns:
            list[GenesisIndividual]: the generated genesis offspring.
        """

        metaPopSize: int = len(metaPopulation)
        genesisPopSize: int = len(genesisPopulation)
        genesisPopPerMetaInd: int = genesisPopSize // metaPopSize
        genesisPopMetaBoundary: int = genesisPopPerMetaInd
        offspring: list[GenesisIndividual] = []

        for metaIndex, _ in enumerate(metaPopulation):
            subOffspring: list[GenesisIndividual] = genesisPopulation[
                metaIndex * genesisPopPerMetaInd : genesisPopMetaBoundary
            ]

            random.shuffle(subOffspring)

            for child1, child2 in zip(subOffspring[::2], subOffspring[1::2]):
                if random.random() < self._genesisCxPb:

                    self._toolbox.genesisMate(child1, child2)

                    del child1.fitness.values
                    del child2.fitness.values
                    child1.id = uuid4()
                    child2.id = uuid4()

            for mutant in subOffspring:
                if random.random() < self._genesisMutPb:
                    self._toolbox.genesisMutate(mutant)

                    del mutant.fitness.values

            offspring.extend(subOffspring)

            genesisPopMetaBoundary += genesisPopPerMetaInd

        return offspring

    def _evaluateGenesisFitness(
        self, metaGen: int, genesisGen: int, genesisPopulation: list[Individual]
    ) -> tuple[list[Individual], list[Individual]]:
        """
        Evaluates the fitness of the genesis individuals.

        Parameters:
            metaGen (int): the current meta-generation.
            genesisGen (int): the current generation.
            genesisPopulation (list[Individual]): the genesis population to evaluate.

        Returns:
            tuple[list[Individual], list[Individual]]: the evaluated genesis population and the qualified individuals
        """

        TUI.appendToSolverLog(
            f"Decoding population for genesis generation {genesisGen} and meta generation {metaGen}."
        )
        populationEG: list[DecodedIndividual] = GenesisUtils.decodePop(
            genesisPopulation, self._topology, self._sfcrs
        )
        TUI.appendToSolverLog(
            f"Population decoded for genesis generation {genesisGen}. Starting evaluation."
        )

        TUI.appendToSolverLog(
            f"Caching surrogate evaluations for genesis generation {genesisGen}."
        )
        if self._objectiveType == POWER:
            HybridEvaluation.cacheForOfflinePowerUsage(
                populationEG,
                self._trafficDesign,
                self._topology,
                genesisGen,
                isAvgOnly=True,
            )
        else:
            HybridEvaluation.cacheForOffline(
                populationEG,
                self._trafficDesign,
                self._topology,
                genesisGen,
                isAvgOnly=True,
            )

        startTime: float = timeit.default_timer()

        TUI.appendToSolverLog(
            f"Evaluating population using surrogate for genesis generation {genesisGen} and meta generation {metaGen}."
        )
        with ProcessPoolExecutor() as executor:
            futures = []

            if self._objectiveType == POWER:
                futures = [
                    executor.submit(
                        HybridEvaluation.evaluationOnSurrogatePowerUsage,
                        ind,
                        genesisGen,
                        self._genesisMaxGen,
                        self._topology,
                        self._trafficDesign,
                        self._maxMemoryDemand,
                    )
                    for ind in populationEG
                ]
            else:
                futures = [
                    executor.submit(
                        HybridEvaluation.evaluationOnSurrogate,
                        ind,
                        genesisGen,
                        self._genesisMaxGen,
                        self._topology,
                        self._trafficDesign,
                        self._maxMemoryDemand,
                    )
                    for ind in populationEG
                ]

            for future in as_completed(futures):
                result: "tuple[int, float, float]" = future.result()
                ind: "Individual" = genesisPopulation[result[0]]
                ind.fitness.values = (result[1], result[2])
        endTime: float = timeit.default_timer()
        TUI.appendToSolverLog(
            f"Finished generation {genesisGen} in {endTime - startTime} seconds."
        )

        hof: tools.ParetoFront = tools.ParetoFront()
        hof.update(genesisPopulation)

        ars = [ind.fitness.values[0] for ind in genesisPopulation]
        latencies = [ind.fitness.values[1] for ind in genesisPopulation]

        self._writeFitnessLog(metaGen, genesisGen, ars, latencies, "surrogate")
        self._writePFLog(metaGen, genesisGen, hof, "surrogate")

        qualifiedIndividuals = [
            ind
            for ind in hof
            if ind.fitness.values[0] >= self._minAR
            and ind.fitness.values[1] <= self._maxSecondObjective
        ]

        TUI.appendToSolverLog(
            f"Qualified Individuals: {len(qualifiedIndividuals)}/{self._minQualInd}"
        )

        if len(qualifiedIndividuals) >= self._minQualInd:
            TUI.appendToSolverLog(
                f"Finished the evolution of weights using surrogate at genesis generation {genesisGen} and meta generation {metaGen}."
            )
            TUI.appendToSolverLog(
                f"Number of qualified individuals: {len(qualifiedIndividuals)}"
            )

            # ---------------------------------------------------------------------------------------------
            # Start the online phase of the hybrid evolution
            # ---------------------------------------------------------------------------------------------

            # If there are more than one individual, select the one with max AR and then min latency.

            if len(qualifiedIndividuals) > 1:
                qualifiedIndividuals.sort(
                    key=lambda ind: (ind.fitness.values[0], -ind.fitness.values[1]),
                    reverse=True,
                )
                qualifiedIndividuals = [qualifiedIndividuals[0]]

            for ind in qualifiedIndividuals:
                del ind.fitness.values

            TUI.appendToSolverLog(
                f"Qualified individual:\n\tRejection rate: {qualifiedIndividuals[0].metaIndividual[0]}\n\tSigma: {qualifiedIndividuals[0].metaIndividual[1]}"
            )

            emHof: tools.ParetoFront = tools.ParetoFront()

            populationEG: "list[DecodedIndividual]" = GenesisUtils.decodePop(
                qualifiedIndividuals, self._topology, self._sfcrs
            )
            HybridEvaluation.cacheForOnline(populationEG, self._trafficDesign)
            for decodedInd in populationEG:
                if self._objectiveType == POWER:
                    ar, latency = HybridEvaluation.evaluationOnEmulatorPowerUsage(
                        decodedInd,
                        self._sfcrs,
                        genesisGen,
                        self._genesisMaxGen,
                        self._sendEGs,
                        self._deleteEGs,
                        self._trafficDesign,
                        self._telemetry,
                        self._topology,
                        self._maxMemoryDemand,
                    )
                else:
                    ar, latency = HybridEvaluation.evaluationOnEmulator(
                        decodedInd,
                        self._sfcrs,
                        genesisGen,
                        self._genesisMaxGen,
                        self._sendEGs,
                        self._deleteEGs,
                        self._trafficDesign,
                        self._trafficGenerator,
                        self._topology,
                        self._maxMemoryDemand,
                    )
                qualifiedIndividuals[decodedInd[0]].fitness.values = (ar, latency)

                for p in genesisPopulation:
                    if p.id == qualifiedIndividuals[decodedInd[0]].id:
                        p.fitness.values = (ar, latency)
                        break

            emHof.update(qualifiedIndividuals)

            ars = [ind.fitness.values[0] for ind in qualifiedIndividuals]
            latencies = [ind.fitness.values[1] for ind in qualifiedIndividuals]

            self._writeFitnessLog(metaGen, genesisGen, ars, latencies, "emulator")
            self._writePFLog(metaGen, genesisGen, emHof, "emulator")

            qualifiedIndividuals = [
                ind
                for ind in emHof
                if ind.fitness.values[0] >= self._minAR
                and ind.fitness.values[1] <= self._maxSecondObjective
            ]

            emMinAR = min(ars)
            emMaxLatency = max(latencies)

            TUI.appendToSolverLog(
                f"Generation {genesisGen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
            )

        return genesisPopulation, qualifiedIndividuals

    def _genesisSelect(
        self,
        metaPopulation: list[Individual],
        genesisParentPopulation: list[GenesisIndividual],
        genesisOffspring: list[GenesisIndividual],
    ) -> list[GenesisIndividual]:
        """
        Selects the genesis population for the next generation.

        Parameters:
            metaPopulation (list[Individual]): the meta-population.
            genesisParentPopulation (list[GenesisIndividual]): the parent genesis population.
            genesisOffspring (list[GenesisIndividual]): the offspring genesis population.

        Returns:
            list[GenesisIndividual]: the selected genesis population for the next generation.
        """

        metaPopSize: int = len(metaPopulation)
        genesisPopSize: int = len(genesisParentPopulation)
        genesisPopPerMetaInd: int = genesisPopSize // metaPopSize
        genesisPopMetaBoundary: int = genesisPopPerMetaInd
        newPop: list[GenesisIndividual] = []

        for metaIndex, _ in enumerate(metaPopulation):
            newPop.extend(
                self._toolbox.select(
                    genesisParentPopulation[
                        metaIndex * genesisPopPerMetaInd : genesisPopMetaBoundary
                    ]
                    + genesisOffspring[
                        metaIndex * genesisPopPerMetaInd : genesisPopMetaBoundary
                    ],
                    k=genesisPopPerMetaInd,
                )
            )
            genesisPopMetaBoundary += genesisPopPerMetaInd

        return newPop

    def evolve(self) -> None:
        """
        Evolves the meta-individuals.

        Returns:
            None
        """

        expStartTime: float = timeit.default_timer()
        self._createLogFiles(
            self._dirName,
            self._experimentName,
            "latency" if self._objectiveType == LATENCY else "power",
        )
        GenesisUtils.init(self._sfcrs, self._topology, self._noOfNeurons, 0.0, 0.0)
        self._initialiseMetaEvolver()

        if (
            not self._retainPopulation
            or len(HierarchicalEvolution._metaPopulation) == 0
        ):
            HierarchicalEvolution._metaPopulation: list[Individual] = (
                self._toolbox.metaPopulation(n=self._metaPopSize)
            )

        if (
            not self._retainPopulation
            or len(HierarchicalEvolution._genesisPopulation) == 0
        ):
            for metaIndividual in HierarchicalEvolution._metaPopulation:
                genesisPopulation: list[GenesisIndividual] = (
                    self._generateGenesisPopulation(metaIndividual)
                )
                HierarchicalEvolution._genesisPopulation.extend(genesisPopulation)

        metaGen: int = 1
        genesisGen: int = 1
        qualifiedIndividuals: list[Individual] = []

        evaluatedPop, qualInd = self._evaluateGenesisFitness(
            metaGen,
            genesisGen,
            cast(list[Individual], HierarchicalEvolution._genesisPopulation),
        )

        qualifiedIndividuals.extend(qualInd)

        HierarchicalEvolution._genesisPopulation = deepcopy(
            cast(list[GenesisIndividual], evaluatedPop)
        )

        HierarchicalEvolution._metaPopulation = self._evaluateMetaFitness(
            HierarchicalEvolution._metaPopulation,
            HierarchicalEvolution._genesisPopulation,
        )

        self._writeMetaLog(metaGen, HierarchicalEvolution._metaPopulation)

        while (
            len(qualifiedIndividuals) < self._minQualInd and metaGen <= self._metaMaxGen
        ):
            metaGen += 1
            metaOffspring: list[Individual] = self._generateMetaOffspring(
                cast(list[Individual], HierarchicalEvolution._metaPopulation),
                self._metaCxPb,
                self._metaMutPb,
            )
            HierarchicalEvolution._genesisPopulation = (
                self._updateMetaIndividualOfGenesisPopulation(
                    metaOffspring,
                    cast(
                        list[GenesisIndividual],
                        HierarchicalEvolution._genesisPopulation,
                    ),
                )
            )

            while (
                len(qualifiedIndividuals) < self._minQualInd
                and genesisGen <= self._genesisMaxGen
            ):
                genesisGen += 1
                genesisOffspring: list[GenesisIndividual] = (
                    self._generateGenesisOffspring(
                        HierarchicalEvolution._genesisPopulation, metaOffspring
                    )
                )
                evaluatedPop, qualInd = self._evaluateGenesisFitness(
                    metaGen, genesisGen, cast(list[Individual], genesisOffspring)
                )

                qualifiedIndividuals.extend(qualInd)
                genesisNewPop: list[GenesisIndividual] = self._genesisSelect(
                    metaOffspring,
                    cast(
                        list[GenesisIndividual],
                        HierarchicalEvolution._genesisPopulation,
                    ),
                    cast(list[GenesisIndividual], genesisOffspring),
                )
                HierarchicalEvolution._genesisPopulation = deepcopy(
                    cast(list[GenesisIndividual], genesisNewPop)
                )

            if len(qualifiedIndividuals) < self._minQualInd:
                metaOffspring = self._evaluateMetaFitness(
                    metaOffspring,
                    cast(
                        list[GenesisIndividual],
                        HierarchicalEvolution._genesisPopulation,
                    ),
                )
                self._writeMetaLog(metaGen, metaOffspring)
                HierarchicalEvolution._metaPopulation = self._selectMetaPopulation(
                    HierarchicalEvolution._metaPopulation,
                    metaOffspring,
                )

        expEndTime: float = timeit.default_timer()

        experimentNames: list[str] = self._experimentName.split("_")

        with open(
            os.path.join(self._experimentDir, "experiment.txt"),
            "w",
            encoding="utf8",
        ) as expFile:
            expFile.write(f"No. of SFCRs: {int(experimentNames[0])}\n")
            expFile.write(f"Traffic Scale: {float(experimentNames[1]) * 10}\n")
            expFile.write(
                f"Traffic Pattern: {'Pattern B' if experimentNames[2] == 'True' else 'Pattern A'}\n"
            )
            expFile.write(f"Link Bandwidth: {experimentNames[3]}\n")
            expFile.write(f"No. of CPUs: {experimentNames[4]}\n")
            expFile.write(f"Time taken: {expEndTime - expStartTime:.2f}\n")
            expFile.write(f"Qualified Individuals: {len(qualifiedIndividuals)}\n")
            for ind in qualifiedIndividuals:
                ind = cast(GenesisIndividual, ind)
                expFile.write(
                    f"Rejection Rate: {ind.metaIndividual[0]}\nSigma: {ind.metaIndividual[1]}\nAR: {ind.fitness.values[0]}\n{'Latency' if self._objectiveType == LATENCY else 'Power'}: {ind.fitness.values[1]}\n"
                )
