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
from algorithms.hybrid.utils.root_evolver import RootEvolver
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
        popSize: int,
        maxGen: int,
        metaCxPb: float,
        genesisCxPb: float,
        metaMutPb: float,
        genesisMutPb: float,
        metaIndPb: float,
        genesisIndPb: float,
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
        dominanceThreshold: float,
        retainPopulation: bool = False,
        rootIndividual: int = -1,
    ) -> None:
        """
        Initializes the hierarchical evolution.

        Parameters:
            popSize (int): the total population size for the hierarchical evolution.
            maxGen (int): the maximum number of generations for the hierarchical evolution.
            metaCxPb (float): the probability of mating two meta-individuals.
            metaMutPb (float): the probability of mutating a meta-individual.
            metaIndPb (float): the independent probability for each attribute to be mutated in a meta-individual.
            genesisIndPb (float): the independent probability for each attribute to be mutated in a genesis individual.
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
            dominanceThreshold (float): the threshold for determining if a Pareto front is dominated by another.
            retainPopulation (bool): specifies if the population should be retained in memory after evolution.
            rootIndividual (int): the index of the root individual to be used in the evolution.

        Returns:
            None
        """

        self._toolbox: base.Toolbox = base.Toolbox()
        self._pfs: str = ""
        self._fitness: str = ""
        self._meta: str = ""
        self._experimentDir: str = ""
        self._popSize: int = popSize
        self._metaCxPb: float = metaCxPb
        self._metaMutPb: float = metaMutPb
        self._metaIndPb: float = metaIndPb
        self._genesisIndPb: float = genesisIndPb
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
        self._dominanceThreshold: float = dominanceThreshold
        self._maxGen: int = maxGen
        self._rootEvolver: RootEvolver = RootEvolver(popSize)
        self._metaPopSize: int = 0
        self._genesisPopSize: int = 0
        self._rootIndividual: int = rootIndividual

    def _computeMetaAndGenesisPopSize(self, root: int) -> tuple[int, int]:
        """
        Computes the meta and genesis population size based on the total population size and the number of meta-individuals.

        Parameters:
            root (int): the number of meta-individuals, which is also the root individual.

        Returns:
            tuple[int, int]: The computed meta and genesis population sizes.
        """

        metaPopSize: int = root
        genesisPopSize: int = self._popSize // metaPopSize

        return metaPopSize, genesisPopSize


    def _calculateParetoDominatedPercentage(self, pf1: tools.ParetoFront, pf2: tools.ParetoFront) -> float:
        """
        Calculates the percentage of individuals in the first Pareto front that are dominated by the second Pareto front.

        Parameters:
            pf1 (tools.ParetoFront): The first Pareto front.
            pf2 (tools.ParetoFront): The second Pareto front.

        Returns:
            float: The percentage of individuals in the first Pareto front that are dominated by the second Pareto front.
        """

        pf2length: int = len(pf2)
        oldCount: int = 0

        for ind2 in pf2:
            for ind1 in pf1:
                if ind2.id == ind1.id:
                    oldCount += 1
                    break

        return (pf2length - oldCount) / pf2length if pf2length > 0 else 0

    def _isParetoDominated(self, pf1: tools.ParetoFront, pf2: tools.ParetoFront) -> bool:
        """
        Determines if the first Pareto front is dominated by the second Pareto front based on a given threshold.

        Parameters:
            pf1 (tools.ParetoFront): The first Pareto front.
            pf2 (tools.ParetoFront): The second Pareto front.

        Returns:
            bool: True if the first Pareto front is dominated by the second Pareto front, False otherwise.
        """

        dominatedPercentage: float = self._calculateParetoDominatedPercentage(pf1, pf2)
        TUI.appendToSolverLog(f"Dominated percentage: {dominatedPercentage}")

        return dominatedPercentage > self._dominanceThreshold

    def _recomposeEvolvers(self, prevRoot: int) -> None:
        """
        Recomposes the evolvers for the hierarchical evolution.

        Parameters:
            prevRoot (int): The previous root individual.

        Returns:
            None
        """

        prevMetaPopSize, prevGenesisPopSize = self._computeMetaAndGenesisPopSize(prevRoot)

        if prevMetaPopSize == self._metaPopSize:
            return

        if self._metaPopSize > prevMetaPopSize:
            TUI.appendToSolverLog("New meta population larger than previous meta population.")
            genesisPopBoundary: int = prevGenesisPopSize

            newGenesisPopulation: list[GenesisIndividual] = []
            for i in range(prevMetaPopSize):
                subGenesisPop: list[GenesisIndividual] = HierarchicalEvolution._genesisPopulation[i * prevGenesisPopSize : genesisPopBoundary]
                selectedGenesisPop: list[GenesisIndividual] = self._toolbox.select(subGenesisPop, k=self._genesisPopSize)
                if len(selectedGenesisPop) < self._genesisPopSize:
                    diff: int = len(selectedGenesisPop) - self._genesisPopSize
                    unSelectedGenesisPop: list[GenesisIndividual] = [ind for ind in subGenesisPop if ind.id not in [selected.id for selected in selectedGenesisPop]]
                    selectedGenesisPop.extend(random.choices(unSelectedGenesisPop, k=abs(diff)))
                newGenesisPopulation.extend(selectedGenesisPop)
                genesisPopBoundary += prevGenesisPopSize
            TUI.appendToSolverLog(f"Pruned GENESIS population to size: {len(newGenesisPopulation)}.")
            HierarchicalEvolution._genesisPopulation = newGenesisPopulation

            for _ in range(self._metaPopSize - prevMetaPopSize):
                metaIndividual: Individual = self._toolbox.metaIndividual()
                genesisPopulation: list[GenesisIndividual] = self._generateGenesisPopulation(metaIndividual)
                TUI.appendToSolverLog(f"Generated new GENESIS population for meta individual: {metaIndividual.id}")
                HierarchicalEvolution._metaPopulation.append(metaIndividual)
                HierarchicalEvolution._genesisPopulation.extend(genesisPopulation)
            TUI.appendToSolverLog(f"Total GENESIS population size after adding new meta individuals: {len(HierarchicalEvolution._genesisPopulation)}.")
        elif self._metaPopSize < prevMetaPopSize:
            TUI.appendToSolverLog("New meta population smaller than previous meta population.")
            selectedMetaPop: list[Individual] = self._toolbox.select(HierarchicalEvolution._metaPopulation, k=self._metaPopSize)
            if len(selectedMetaPop) < self._metaPopSize:
                diff: int = len(selectedMetaPop) - self._metaPopSize
                unSelectedMetaPop: list[Individual] = [ind for ind in HierarchicalEvolution._metaPopulation if ind.id not in [selected.id for selected in selectedMetaPop]]
                selectedMetaPop.extend(random.choices(unSelectedMetaPop, k=abs(diff)))
            selectedGenesisPop: list[GenesisIndividual] = []
            genesisPopBoundary: int = prevGenesisPopSize
            for i, metaInd in enumerate(HierarchicalEvolution._metaPopulation):
                if len([selectedMetaInd for selectedMetaInd in selectedMetaPop if selectedMetaInd.id == metaInd.id]) == 1:
                    selectedGenesisPop.extend(HierarchicalEvolution._genesisPopulation[i * prevGenesisPopSize : genesisPopBoundary])
                genesisPopBoundary += prevGenesisPopSize

            HierarchicalEvolution._metaPopulation = [metaInd for metaInd in HierarchicalEvolution._metaPopulation if metaInd.id in [selectedMetaInd.id for selectedMetaInd in selectedMetaPop]]
            HierarchicalEvolution._genesisPopulation = selectedGenesisPop

            TUI.appendToSolverLog(f"Pruned meta population to size: {len(HierarchicalEvolution._metaPopulation)}.")
            TUI.appendToSolverLog(f"Total GENESIS population size after pruning meta individuals: {len(HierarchicalEvolution._genesisPopulation)}.")
            newGenesisPopulation: list[GenesisIndividual] = []
            genesisPopBoundary: int = prevGenesisPopSize
            for i, metaInd in enumerate(HierarchicalEvolution._metaPopulation):
                subGenesisPop: list[GenesisIndividual] = HierarchicalEvolution._genesisPopulation[i * prevGenesisPopSize : genesisPopBoundary]

                for _ in range(self._genesisPopSize - prevGenesisPopSize):
                    selectedGenesisInd: GenesisIndividual = cast(GenesisIndividual, GenesisUtils.generateRandomGenesisIndividual(GenesisIndividual, self._topology, self._sfcrs))
                    selectedGenesisInd.metaIndividual = metaInd
                    subGenesisPop.append(selectedGenesisInd)

                newGenesisPopulation.extend(subGenesisPop)
                genesisPopBoundary += prevGenesisPopSize

            HierarchicalEvolution._genesisPopulation = newGenesisPopulation
            TUI.appendToSolverLog(f"Recomposed GENESIS population to size: {len(HierarchicalEvolution._genesisPopulation)}.")


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
        self._root = os.path.join(experimentDir, "root.csv")

        with open(self._pfs, "w") as pfFile:
            pfFile.write(
                f"method,generation,root_generation,meta_generation,genesis_generation,ar,{secondObjective}\n"
            )
        with open(self._fitness, "w") as fitnessFile:
            fitnessFile.write(
                f"method,generation,root_generation,meta_generation,genesis_generation,average_ar,max_ar,min_ar,average_{secondObjective},max_{secondObjective},min_{secondObjective}\n"
            )
        with open(self._meta, "w") as metaFile:
            metaFile.write(
                "root_generation,meta_generation,average_rejection_rate,max_rejection_rate,min_rejection_rate,average_sigma,max_sigma,min_sigma,dominance\n"
            )
        with open(self._root, "w") as rootFile:
            rootFile.write("root_generation,meta_pop_size,genesis_pop_size,dominance\n")

    def _writeFitnessLog(
        self,
        gen: int,
        metaGen: int,
        genesisGen: int,
        rootGen: int,
        ars: list[float],
        secondObjectives: list[float],
        method: str
    ) -> None:
        """
        Writes the fitness log for the hierarchical evolution.

        Parameters:
            gen (int): the current generation.
            metaGen (int): the current meta-generation.
            genesisGen (int): the current genesis generation.
            rootGen (int): the current root generation.
            ars (list[float]): the list of ARs of the population.
            secondObjectives (list[float]): the list of second objective values of the population.
            method (str): the method name.

        Returns:
            None
        """

        with open(self._fitness, "a") as fitnessFile:
            fitnessFile.write(
                f"{method},{gen},{rootGen},{metaGen},{genesisGen},{np.mean(ars)},{max(ars)},{min(ars)},{np.mean(secondObjectives)},{max(secondObjectives)},{min(secondObjectives)}\n"
            )

    def _writePFLog(
        self, gen: int, metaGen: int, genesisGen: int, rootGen: int, hof: tools.ParetoFront, method: str
    ) -> None:
        """
        Writes the Pareto front log for the hierarchical evolution.

        Parameters:
            gen (int): the current generation.
            metaGen (int): the current meta-generation.
            genesisGen (int): the current genesis generation.
            rootGen (int): the current root generation.
            hof (tools.ParetoFront): the Pareto front of the population.
            method (str): the method name.

        Returns:
            None
        """

        with open(self._pfs, "a") as pfFile:
            for individual in hof:
                pfFile.write(
                    f"{method},{gen},{rootGen},{metaGen},{genesisGen},{individual.fitness.values[0]},{individual.fitness.values[1]}\n"
                )

    def _writeMetaLog(self, rootGen: int, metaGen: int, metaPopulation: list[Individual], dominance: float) -> None:
        """
        Writes the meta log for the hierarchical evolution.

        Parameters:
            rootGen (int): the current root generation.
            metaGen (int): the current meta-generation.
            metaPopulation (list[Individual]): the meta-population.
            dominance (float): the dominance value.

        Returns:
            None
        """

        with open(self._meta, "a") as metaFile:
            rejectionRates: list[float] = [ind[0] for ind in metaPopulation]
            sigmas: list[float] = [ind[1] for ind in metaPopulation]
            metaFile.write(
                f"{rootGen},{metaGen},{np.mean(rejectionRates)},{max(rejectionRates)},{min(rejectionRates)},{np.mean(sigmas)},{max(sigmas)},{min(sigmas)},{dominance}\n"
            )

    def _writeRootLog(self, rootGen: int, dominance: float) -> None:
        """
        Writes the root log for the hierarchical evolution.

        Parameters:
            rootGen (int): the current root generation.
            dominance (float): the dominance value.

        Returns:
            None
        """

        with open(self._root, "a") as rootFile:
            rootFile.write(f"{rootGen},{self._metaPopSize},{self._genesisPopSize},{dominance}\n")


    def _generateHyperParameters(self) -> list[float]:
        """
        Generates random hyper-parameters.

        Returns:
            list[float]: the generated hyper-parameters.
        """

        return [
            abs(round(random.uniform(0, 1), 2)),
            round(random.uniform(0, 10), 2),
        ]

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

                if child[0] < 0:
                    child[0] = 0

                if child[0] > 1:
                    child[0] = 1

                if child[1] < 0:
                    child[1] = 0


                if child2[0] < 0:
                    child2[0] = 0

                if child2[0] > 1:
                    child2[0] = 1

                if child2[1] < 0:
                    child2[1] = 0

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

    def _performGAOperationsGenesis(
        self, gen: int, rootGen: int, metaGen: int, genesisGen: int, genesisPopulation: list[Individual], parentPopulation: list[Individual], parentPopEG: list[DecodedIndividual]
    ) -> tuple[list[Individual], list[Individual], list[DecodedIndividual]]:
        """
        Evaluates the fitness of the genesis individuals.

        Parameters:
            gen (int): the current generation.
            rootGen (int): the current root generation.
            metaGen (int): the current meta-generation.
            genesisGen (int): the current generation.
            genesisPopulation (list[Individual]): the genesis population to evaluate.
            parentPopulation (list[Individual]): the parent genesis population.
            parentPopEG (list[DecodedIndividual]): the decoded parent population.

        Returns:
            tuple[list[Individual], list[Individual], list[DecodedIndividual]]: the evaluated genesis population, the qualified individuals, and the decoded genesis population.
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
                        100,
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
                        100,
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
            f"Finished meta generation {metaGen} and genesis generation {genesisGen} in {endTime - startTime} seconds."
        )

        genesisNewPop: list[GenesisIndividual] = []
        if len(parentPopulation) > 0:
            genesisNewPop: list[GenesisIndividual] = self._genesisSelect(
                    cast(
                        list[GenesisIndividual],
                        parentPopulation,
                    ),
                    cast(list[GenesisIndividual], genesisPopulation),
                )
        else:
            genesisNewPop = cast(list[GenesisIndividual], genesisPopulation)

        popEG: list[DecodedIndividual] = GenesisUtils.extractDecodedIndividuals(
            cast(list[Individual], genesisNewPop), parentPopEG + populationEG
        )
        ars = [ind.fitness.values[0] for ind in genesisNewPop]
        latencies = [ind.fitness.values[1] for ind in genesisNewPop]
        hof: tools.ParetoFront = tools.ParetoFront()
        hof.update(genesisNewPop)

        self._writeFitnessLog(gen, metaGen, genesisGen, rootGen, ars, latencies, "surrogate")
        self._writePFLog(gen, metaGen, genesisGen, rootGen, hof, "surrogate")

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

            populationEG: "list[DecodedIndividual]" = GenesisUtils.extractDecodedIndividuals(
                qualifiedIndividuals, popEG
            )
            HybridEvaluation.cacheForOnline(populationEG, self._trafficDesign)
            for i, decodedInd in enumerate(populationEG):
                if self._objectiveType == POWER:
                    ar, latency = HybridEvaluation.evaluationOnEmulatorPowerUsage(
                        decodedInd,
                        self._sfcrs,
                        gen,
                        self._maxGen,
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
                        gen,
                        self._maxGen,
                        self._sendEGs,
                        self._deleteEGs,
                        self._trafficDesign,
                        self._trafficGenerator,
                        self._topology,
                        self._maxMemoryDemand,
                    )
                qualifiedIndividuals[i].fitness.values = (ar, latency)

                for p in genesisNewPop:
                    if p.id == qualifiedIndividuals[i].id:
                        p.fitness.values = (ar, latency)
                        break

            emHof.update(qualifiedIndividuals)

            ars = [ind.fitness.values[0] for ind in qualifiedIndividuals]
            latencies = [ind.fitness.values[1] for ind in qualifiedIndividuals]

            self._writeFitnessLog(gen, metaGen, genesisGen, rootGen, ars, latencies, "emulator")
            self._writePFLog(gen, metaGen, genesisGen, rootGen, emHof, "emulator")

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

        return cast(list[Individual], genesisNewPop), qualifiedIndividuals, popEG

    def _genesisSelect(
        self,
        genesisParentPopulation: list[GenesisIndividual],
        genesisOffspring: list[GenesisIndividual],
    ) -> list[GenesisIndividual]:
        """
        Selects the genesis population for the next generation.

        Parameters:
            genesisParentPopulation (list[GenesisIndividual]): the parent genesis population.
            genesisOffspring (list[GenesisIndividual]): the offspring genesis population.

        Returns:
            list[GenesisIndividual]: the selected genesis population for the next generation.
        """

        metaPopSize: int = self._metaPopSize
        genesisPopSize: int = len(genesisParentPopulation)
        genesisPopPerMetaInd: int = genesisPopSize // metaPopSize
        genesisPopMetaBoundary: int = genesisPopPerMetaInd
        newPop: list[GenesisIndividual] = []

        for metaIndex in range(metaPopSize):
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

    def _getWorstFitness(self, population: list[Individual]) -> tuple[float, float]:
        """
        Gets the worst fitness of the population.

        Parameters:
            population (list[Individual]): the population to get the worst fitness from.

        Returns:
            tuple[float, float]: the worst fitness of the population.
        """

        worstAR: float = min([ind.fitness.values[0] for ind in population])
        worstLatency: float = max([ind.fitness.values[1] for ind in population])

        return worstAR, worstLatency

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
            or RootEvolver.getRootIndividual() == -1
        ):
            TUI.appendToSolverLog("Root not retained. Generating random root.")
            RootEvolver.setRootIndividual(self._rootEvolver.generateRandomRoot())

        if self._rootIndividual != -1:
            TUI.appendToSolverLog(f"Root predefined. Using root: {self._rootIndividual}.")
            RootEvolver.setRootIndividual(self._rootIndividual)

        self._rootEvolver.addRootToExploredRoot(RootEvolver.getRootIndividual())

        TUI.appendToSolverLog(f"Root is: {RootEvolver.getRootIndividual()}.")
        self._metaPopSize, self._genesisPopSize = self._computeMetaAndGenesisPopSize(RootEvolver.getRootIndividual())
        TUI.appendToSolverLog(f"Meta pop size: {self._metaPopSize}, Genesis pop size: {self._genesisPopSize}")

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
        gen: int = 0
        rootGen: int = 0
        qualifiedIndividuals: list[Individual] = []

        rootPF: tools.ParetoFront = tools.ParetoFront()
        metaPF: tools.ParetoFront = tools.ParetoFront()
        genesisPF: tools.ParetoFront = tools.ParetoFront()
        rootRadius: float = 0.0
        shouldMetaGenContinue: bool = True
        shouldGenesisGenContinue: bool = True
        prevRootIndividual: int = RootEvolver.getRootIndividual()
        hyperVolumeReferencePoint: tuple[float, float] = (0.0, 0.0)

        while (
            len(qualifiedIndividuals) < self._minQualInd and gen < self._maxGen
        ):
            metaGen = 0
            genesisGen = 0
            rootGen += 1
            gen += 1
            evaluatedPop, qualInd, popEG = self._performGAOperationsGenesis(
                gen,
                rootGen,
                metaGen,
                genesisGen,
                cast(list[Individual], HierarchicalEvolution._genesisPopulation),
                [],
                []
            )

            qualifiedIndividuals.extend(qualInd)

            HierarchicalEvolution._genesisPopulation = deepcopy(
                cast(list[GenesisIndividual], evaluatedPop)
            )

            HierarchicalEvolution._metaPopulation = self._evaluateMetaFitness(
                HierarchicalEvolution._metaPopulation,
                HierarchicalEvolution._genesisPopulation,
            )

            newRootPF: tools.ParetoFront = tools.ParetoFront()
            newMetaPF: tools.ParetoFront = tools.ParetoFront()
            newGenesisPF: tools.ParetoFront = tools.ParetoFront()
            newRootPF.update(HierarchicalEvolution._genesisPopulation)
            newMetaPF.update(HierarchicalEvolution._genesisPopulation)
            newGenesisPF.update(HierarchicalEvolution._genesisPopulation)
            rootDominance: float = self._calculateParetoDominatedPercentage(rootPF, newRootPF)
            metaDominance: float = self._calculateParetoDominatedPercentage(metaPF, newMetaPF)
            self._writeMetaLog(rootGen, metaGen, HierarchicalEvolution._metaPopulation, metaDominance)
            self._writeRootLog(rootGen, rootDominance)
            rootPF = newRootPF
            metaPF = newMetaPF
            genesisPF = newGenesisPF

            while len(qualifiedIndividuals) < self._minQualInd and shouldMetaGenContinue and gen < self._maxGen:
                metaGen += 1
                metaPopulation: list[Individual] = deepcopy(
                    cast(list[Individual], HierarchicalEvolution._metaPopulation)
                )
                metaOffspring: list[Individual] = []

                if self._metaPopSize > 1:
                    metaOffspring = self._generateMetaOffspring(
                        cast(list[Individual], metaPopulation),
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
                else:
                    metaOffspring = deepcopy(metaPopulation)

                genesisGen = 0
                while (
                    len(qualifiedIndividuals) < self._minQualInd
                    and shouldGenesisGenContinue and gen < self._maxGen
                ):
                    gen += 1
                    genesisGen += 1

                    genesisPopulation: list[GenesisIndividual] = deepcopy(
                        cast(list[GenesisIndividual], HierarchicalEvolution._genesisPopulation)
                    )

                    genesisOffSpring: list[GenesisIndividual] = []

                    if self._genesisPopSize > 1:
                        genesisOffSpring = (
                            self._generateGenesisOffspring(
                                genesisPopulation, metaOffspring
                            )
                        )
                    else:
                        genesisOffSpring = deepcopy(genesisPopulation)

                    evaluatedPop, qualInd, popEG = self._performGAOperationsGenesis(
                        gen, rootGen, metaGen, genesisGen, cast(list[Individual], genesisOffSpring), cast(list[Individual], HierarchicalEvolution._genesisPopulation), popEG
                    )

                    if gen == 1:
                        hyperVolumeReferencePoint = self._getWorstFitness(evaluatedPop)

                    qualifiedIndividuals.extend(qualInd)
                    newGenesisPF: tools.ParetoFront = tools.ParetoFront()
                    newGenesisPF.update(evaluatedPop)
                    shouldGenesisGenContinue = self._isParetoDominated(
                        genesisPF, newGenesisPF
                    )
                    genesisPF = newGenesisPF
                    HierarchicalEvolution._genesisPopulation = deepcopy(
                        cast(list[GenesisIndividual], evaluatedPop)
                    )
                if len(qualifiedIndividuals) >= self._minQualInd or gen >= self._maxGen:
                    break

                TUI.appendToSolverLog("Exiting GENESIS evolution and moving to the next generation of meta.")
                metaOffspring = self._evaluateMetaFitness(
                    metaOffspring,
                    cast(
                        list[GenesisIndividual],
                        HierarchicalEvolution._genesisPopulation,
                    ),
                )

                HierarchicalEvolution._metaPopulation = self._selectMetaPopulation(
                    HierarchicalEvolution._metaPopulation,
                    metaOffspring,
                )
                newMetaPF: tools.ParetoFront = tools.ParetoFront()
                newMetaPF.update(HierarchicalEvolution._genesisPopulation)
                shouldMetaGenContinue = self._isParetoDominated(
                    metaPF, newMetaPF
                ) or self._rootIndividual == -1
                dominance: float = self._calculateParetoDominatedPercentage(metaPF, newMetaPF)
                self._writeMetaLog(rootGen, metaGen, metaOffspring, dominance)
                metaPF = newMetaPF
                shouldGenesisGenContinue = True

            if len(qualifiedIndividuals) >= self._minQualInd or gen >= self._maxGen:
                break
            TUI.appendToSolverLog("Exiting meta evolution and moving to the next generation of root.")
            newRootPF: tools.ParetoFront = tools.ParetoFront()
            newRootPF.update(HierarchicalEvolution._genesisPopulation)
            isCurrentRootDominant = self._isParetoDominated(
                rootPF, newRootPF
            )
            dominance: float = self._calculateParetoDominatedPercentage(rootPF, newRootPF)
            self._writeRootLog(rootGen, dominance)
            currentRootIndividual = RootEvolver.getRootIndividual()
            if isCurrentRootDominant:
                prevRootIndividual = RootEvolver.getRootIndividual()
                rootRadius = 1 - dominance
                RootEvolver.setRootIndividual(self._rootEvolver.selectRootNeighbour(
                    RootEvolver.getRootIndividual(), rootRadius
                ))
            else:
                TUI.appendToSolverLog("Root is not dominated by previous root.")
                RootEvolver.setRootIndividual(self._rootEvolver.selectRootNeighbour(
                    prevRootIndividual, rootRadius
                ))
            self._rootEvolver.addRootToExploredRoot(RootEvolver.getRootIndividual())
            TUI.appendToSolverLog(f"New root: {RootEvolver.getRootIndividual()}")
            self._metaPopSize, self._genesisPopSize = self._computeMetaAndGenesisPopSize(RootEvolver.getRootIndividual())
            TUI.appendToSolverLog(f"Meta pop size: {self._metaPopSize}, Genesis pop size: {self._genesisPopSize}")
            self._recomposeEvolvers(currentRootIndividual)

            rootPF = newRootPF
            shouldMetaGenContinue = True

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
            expFile.write(
                f"Meta Pop Size: {self._metaPopSize}\n"
            )
            expFile.write(
                f"Genesis Pop Size: {self._genesisPopSize}\n"
            )
            for ind in qualifiedIndividuals:
                ind = cast(GenesisIndividual, ind)
                expFile.write(
                    f"Rejection Rate: {ind.metaIndividual[0]}\nSigma: {ind.metaIndividual[1]}\nAR: {ind.fitness.values[0]}\n{'Latency' if self._objectiveType == LATENCY else 'Power'}: {ind.fitness.values[1]}\n"
                )
