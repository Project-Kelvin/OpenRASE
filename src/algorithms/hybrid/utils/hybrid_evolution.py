"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import os
import random
import timeit
from typing import Callable, Tuple, Type, Union, cast
from uuid import UUID, uuid4
from algorithms.mak_ga.mak_ga_utils import MakGAUtils
from deap import base, tools
import numpy as np
from shared.models.sfc_request import SFCRequest
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from shared.utils.config import getConfig
from algorithms.hybrid.constants.genesis_objective import LATENCY, POWER
from algorithms.hybrid.models.individuals import Individual
from algorithms.hybrid.utils.genesis import GenesisUtils
from algorithms.models.embedding import DecodedIndividual
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from mano.telemetry import Telemetry
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI

MAX_MEMORY_DEMAND: int = 10
MAX_LATENCY: int = 100
MAX_POWER: int = 300
MIN_AR: float = 0.95
MIN_QUAL_IND: int = 1
NGEN: int = 100
TIME_LIMIT: int = 1 # in hours

DecodePop = Callable[
    [list[Individual], Topology, list[SFCRequest]], list[DecodedIndividual]
]
GenerateRandomIndividual = Callable[[Type[Individual], Topology, list[SFCRequest]], Individual]
Crossover = Callable[
    [Individual, Individual],
    Tuple[Individual, Individual],
]
Mutate = Callable[[Individual, float], Individual]
RejectVNF = Callable[[Individual, float], Individual]


class HybridEvolution:
    """
    This class handles the hybrid evolution process, which includes both offline and online phases.
    It uses a surrogate model for the offline phase and an emulator for the online phase.
    """

    _population: list[Individual] = []

    def __init__(
        self,
        experimentName: str,
        decodePop: DecodePop,
        generateRandomIndividual: GenerateRandomIndividual,
        crossover: Crossover,
        mutate: Mutate,
        individualContainer: Type[Individual],
        mutPb: float,
        cxpPb: float,
        indPb: float,
        evaluateOnline: bool = True,
        evaluateOffline: bool = True,
        retrain: bool = False,
        rejectVNF: Union[RejectVNF, None] = None,
        rejectionRate: float = 0.05,
        useGAHAOffline: bool = False,
        finalValidation: bool = False,
    ):
        """
        Initializes the HybridEvolution class.

        Parameters:
            experimentName (str): the name of the experiment.
            decodePop (DecodePop): the function to decode the population.
            generateRandomIndividual (GenerateRandomIndividual): the function to generate a random individual.
            crossover (Crossover): the crossover function.
            mutate (Mutate): the mutation function.
            individualContainer (Type[Individual]): the class type for individuals.
            mutPb (float): the mutation probability.
            cxpPb (float): the crossover probability.
            indPb (float): the individual mutation probability.
            evaluateOnline (bool): whether to evaluate the solution online.
            evaluateOffline (bool): whether to evaluate the solution offline.
            retrain (bool): Specifies if BENNS should be retrained.
            rejectVNF (Union[RejectVNF, None]): the function to reject a VNF.
            rejectionRate (float): the probability of a VNF being deployed on a host.
            useGAHAOffline (bool): use GAHA's offline evaluator.
            finalValidation (bool): whether to validate the best final solution irrespective of convergence.

        Returns:
            None
        """

        self._decodePop: DecodePop = decodePop
        self._generateRandomIndividual: GenerateRandomIndividual = (
            generateRandomIndividual
        )
        self._crossover: Crossover = crossover
        self._mutate: Mutate = mutate
        self._rejectVNF: Union[RejectVNF, None] = rejectVNF
        self._toolbox: base.Toolbox = base.Toolbox()
        self._artifactsDir: str = os.path.join(
            getConfig()["repoAbsolutePath"], "artifacts", "experiments", experimentName
        )
        self._individualContainer: Type[Individual] = individualContainer
        self._mutPb: float = mutPb
        self._indPb: float = indPb
        self._cxpPb: float = cxpPb
        self._evaluateOnline: bool = evaluateOnline
        self._evaluateOffline: bool = evaluateOffline
        self._retrain: bool = retrain
        self._rejectionRate: float = rejectionRate
        self._useGAHAOffline: bool = useGAHAOffline
        self._finalValidation: bool = finalValidation
        self._decodedPop: dict[UUID, DecodedIndividual] = {}

    def _select(
        self,
        offspring: "list[Individual]",
        pop: "list[Individual]",
        popSize: int,
        hof: tools.ParetoFront,
    ) -> "Tuple[list[Individual], tools.ParetoFront]":
        """
        Selection function.

        Parameters:
            offspring (list[Individual]): the offspring.
            pop (list[Individual]): the population.
            popSize (int): the population size.
            hof (tools.ParetoFront): the hall of fame.

        Returns:
            Tuple[list[Individual], tools.ParetoFront]: the population and the hall of fame.
        """

        pop[:] = self._toolbox.select(pop + offspring, k=popSize)

        hof.update(pop)

        return pop, hof

    def _writeData(
        self,
        gen: int,
        ars: "list[float]",
        latencies: "list[float]",
        method: str,
        dir: str,
    ) -> None:
        """
        Writes the data to the file.

        Parameters:
            gen (int): the generation.
            ars (list[float]): the acceptance ratios.
            latencies (list[float]): the latencies.
            method (str): the method used.
            dir (str): the directory to write the data to.

        Returns:
            None
        """

        with open(
            f"{dir}/data.csv",
            "a",
            encoding="utf8",
        ) as dataFile:
            dataFile.write(
                f"{method}, {gen}, {np.mean(ars)}, {max(ars)}, {min(ars)}, {np.mean(latencies)}, {max(latencies)}, {min(latencies)}\n"
            )

    def _writePFs(
        self, gen: int, hof: tools.ParetoFront, method: str, dir: str
    ) -> None:
        """
        Writes the Pareto Fronts to the file.

        Parameters:
            gen (int): the generation.
            hof (tools.ParetoFront): the hall of fame.
            method (str): the method used.
            dir (str): the directory to write the data to.

        Returns:
            None
        """

        TUI.appendToSolverLog(f"Writing Pareto Fronts for generation {gen}.")
        for ind in hof:
            with open(
                f"{dir}/pfs.csv",
                "a",
                encoding="utf8",
            ) as pfFile:
                pfFile.write(
                    f"{method}, {gen}, {ind.fitness.values[1]}, {ind.fitness.values[0]}\n"
                )

    def _generateOffspring(
        self,
        pop: "list[Individual]"
    ) -> "list[Individual]":
        """
        Generate offspring from the population.

        Parameters:
            pop (list[Individual]): the population.

        Returns:
            offspring (list[Individual]): the offspring.
        """

        offspring: "list[Individual]" = list(map(self._toolbox.clone, pop))
        random.shuffle(offspring)


        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self._cxpPb:

                self._toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values
                child1.id = uuid4()
                child2.id = uuid4()

        for mutant in offspring:
            if random.random() < self._mutPb:
                self._toolbox.mutate(mutant)

                del mutant.fitness.values

            if self._rejectVNF:
                self._rejectVNF(mutant, self._rejectionRate)

        return offspring

    def _decodePopulation(self, pop: list[Individual], topology: Topology, sfcrs: list[SFCRequest]) -> list[DecodedIndividual]:
        """
        Decodes population by looking up the decoded pop dictionary and using the decodePop function.

        Parameters:
            pop (list[Individual]): the encoded population.
            topology (Topology): the network topology.
            sfcrs (list[SFCRequest]): the SFCR requests to embed.

        Returns:
            decodedPop (list[DecodedIndividual]): the decoded population.
        """

        newDecodedPop: list[DecodedIndividual] = self._decodePop(pop, topology, sfcrs)
        for i, ind in enumerate(newDecodedPop):
            if ind[5] in self._decodedPop:
                reIndexedDecodedPop: DecodedIndividual = cast(DecodedIndividual, (i, self._decodedPop[ind[5]][1], self._decodedPop[ind[5]][2], self._decodedPop[ind[5]][3], self._decodedPop[ind[5]][4], self._decodedPop[ind[5]][5]))
                newDecodedPop[i] = reIndexedDecodedPop
            else:
                self._decodedPop[ind[5]] = ind

        return newDecodedPop


    def _performGeneticOperation(
        self,
        parent: list[Individual],
        pop: list[Individual],
        topology: Topology,
        fgrs: list[SFCRequest],
        trafficDesign: list[TrafficDesign],
        dirName: str,
        scoresDir: str,
        gen: int,
        ngen: int,
        maxMemoryDemand: float,
        minAR: float,
        maxObjective: float,
        minQualifiedInds: int,
        popSize: int,
        trafficGenerator: TrafficGenerator,
        telemetry: Telemetry,
        sendEGs: "Callable[[list[EmbeddingGraph]], None]",
        deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
        hof: tools.ParetoFront,
        type: str = LATENCY,
    ) -> "tuple[list[Individual], list[Individual]]":
        """
        Perform the genetic operation.

        Parameters:
            parent (list[Individual]): the parent population.
            pop (list[Individual]): the current population.
            topology (Topology): the topology.
            fgrs (list[SFCRequest]): the SFC Requests.
            trafficDesign (list[TrafficDesign]): the traffic design.
            dirName (str): the directory name to save results.
            scoresDir (str): the directory name for scores.
            gen (int): the current generation number.
            ngen (int): the total number of generations.
            maxMemoryDemand (float): maximum memory demand allowed.
            minAR (float): minimum acceptance ratio required.
            maxLatency (float): maximum latency allowed.
            minQualifiedInds (int): minimum number of qualified individuals required.
            popSize (int): size of the population.
            trafficGenerator (TrafficGenerator): traffic generator instance.
            telemetry (Telemetry): telemetry instance.
            sendEGs (Callable[[list[EmbeddingGraph]], None]): function to send Embedding Graphs.
            deleteEGs (Callable[[list[EmbeddingGraph]], None]): function to delete Embedding Graphs.
            hof (tools.ParetoFront): hall of fame for storing best individuals.
            type (str): the optimisation objective type.

        Returns:
            tuple[list[Individual], list[Individual]]: the updated population, and qualified individuals.
        """

        TUI.appendToSolverLog(f"Decoding population for generation {gen}.")
        populationEG: "list[DecodedIndividual]" = self._decodePopulation(pop, topology, fgrs)
        TUI.appendToSolverLog(
            f"Population decoded for generation {gen}. Starting evaluation."
        )

        makGAUtils: Union[MakGAUtils, None] = None

        if self._evaluateOffline:
            TUI.appendToSolverLog(f"Caching surrogate evaluations for generation {gen}.")
            if type == POWER:
                HybridEvaluation.cacheForOfflinePowerUsage(
                    populationEG, trafficDesign, topology, gen, isAvgOnly=True
                )
            elif self._useGAHAOffline:
                makGAUtils = MakGAUtils(topology, trafficDesign[0], fgrs)
                makGAUtils.cacheDemand(populationEG)
            else:
                HybridEvaluation.cacheForOffline(
                    populationEG, trafficDesign, topology, gen, isAvgOnly=True
                )
                # HybridEvaluation.saveCachedLatency(
                #     os.path.join(dirName, scoresDir, f"gen_{gen}.csv")
                # )

        startTime: float = timeit.default_timer()

        TUI.appendToSolverLog(
            f"Evaluating population using surrogate for generation {gen}."
        )

        if self._evaluateOffline:
            with ProcessPoolExecutor() as executor:
                futures = []

                if type == POWER:
                    futures = [
                        executor.submit(
                            HybridEvaluation.evaluationOnSurrogatePowerUsage,
                            ind,
                            gen,
                            ngen,
                            topology,
                            trafficDesign,
                            maxMemoryDemand,
                        )
                        for ind in populationEG
                    ]
                elif self._useGAHAOffline and makGAUtils is not None:
                    futures = [
                        executor.submit(
                            makGAUtils.getTotalDelay,
                            ind,
                        )
                        for ind in populationEG
                    ]
                else:
                    futures = [
                        executor.submit(
                            HybridEvaluation.evaluationOnSurrogate,
                            ind,
                            gen,
                            ngen,
                            topology,
                            trafficDesign,
                            maxMemoryDemand,
                        )
                        for ind in populationEG
                    ]

                for future in as_completed(futures):
                    result: "tuple[int, float, float]" = future.result()
                    ind: "Individual" = pop[result[0]]
                    ind.fitness.values = (result[1], result[2])
        else:
            for i, decodedInd in enumerate(populationEG):
                if type == POWER:
                    ar, latency = HybridEvaluation.evaluationOnEmulatorPowerUsage(
                        decodedInd,
                        fgrs,
                        gen,
                        ngen,
                        sendEGs,
                        deleteEGs,
                        trafficDesign,
                        telemetry,
                        topology,
                        maxMemoryDemand,
                    )
                else:
                    ar, latency = HybridEvaluation.evaluationOnEmulator(
                        decodedInd,
                        fgrs,
                        gen,
                        ngen,
                        sendEGs,
                        deleteEGs,
                        trafficDesign,
                        trafficGenerator,
                        topology,
                        maxMemoryDemand,
                        self._retrain,
                    )
                pop[i].fitness.values = (ar, latency)

        endTime: float = timeit.default_timer()
        TUI.appendToSolverLog(
            f"Finished generation {gen} in {endTime - startTime} seconds."
        )
        if len(parent) > 0:
            pop, hof = self._select(pop, parent, popSize, hof)
        else:
            hof.update(pop)

        ars = [ind.fitness.values[0] for ind in pop]
        latencies = [ind.fitness.values[1] for ind in pop]

        self._writeData(gen, ars, latencies, "surrogate", dirName)
        self._writePFs(gen, hof, "surrogate", dirName)

        maxSecondObjective: float = maxObjective

        qualifiedIndividuals = [
            ind
            for ind in hof
            if ind.fitness.values[0] >= minAR and ind.fitness.values[1] <= maxSecondObjective
        ]

        TUI.appendToSolverLog(
            f"Qualified Individuals: {len(qualifiedIndividuals)}/{minQualifiedInds}"
        )

        if self._finalValidation and len(qualifiedIndividuals) == 0 and gen == ngen:
            qualifiedIndividuals = list(hof)

        if len(qualifiedIndividuals) >= minQualifiedInds and self._evaluateOffline:
            TUI.appendToSolverLog(
                f"Finished the evolution of weights using surrogate at generation {gen}."
            )
            TUI.appendToSolverLog(
                f"Number of qualified individuals: {len(qualifiedIndividuals)}"
            )

            # ---------------------------------------------------------------------------------------------
            # Start the online phase of the hybrid evolution
            # ---------------------------------------------------------------------------------------------

            if self._evaluateOnline:
                # If there are more than one individual, select the one with max AR and then min latency.

                if len(qualifiedIndividuals) > 1:
                    qualifiedIndividuals.sort(
                        key=lambda ind: (ind.fitness.values[0], -ind.fitness.values[1]),
                        reverse=True,
                    )
                    qualifiedIndividuals = [qualifiedIndividuals[0]]

                for ind in qualifiedIndividuals:
                    del ind.fitness.values

                emHof = tools.ParetoFront()

                populationEG = self._decodePopulation(qualifiedIndividuals, topology, fgrs)
                HybridEvaluation.cacheForOnline(populationEG, trafficDesign)
                for i, decodedInd in enumerate(populationEG):
                    if type == POWER:
                        ar, latency = HybridEvaluation.evaluationOnEmulatorPowerUsage(
                            decodedInd,
                            fgrs,
                            gen,
                            ngen,
                            sendEGs,
                            deleteEGs,
                            trafficDesign,
                            telemetry,
                            topology,
                            maxMemoryDemand,
                        )
                    else:
                        ar, latency = HybridEvaluation.evaluationOnEmulator(
                            decodedInd,
                            fgrs,
                            gen,
                            ngen,
                            sendEGs,
                            deleteEGs,
                            trafficDesign,
                            trafficGenerator,
                            topology,
                            maxMemoryDemand,
                            self._retrain,
                        )
                    qualifiedIndividuals[i].fitness.values = (ar, latency)

                    for p in pop:
                        if p.id == qualifiedIndividuals[i].id:
                            p.fitness.values = (ar, latency)
                            break

                emHof.update(qualifiedIndividuals)

                ars = [ind.fitness.values[0] for ind in qualifiedIndividuals]
                latencies = [ind.fitness.values[1] for ind in qualifiedIndividuals]

                self._writeData(gen + 1, ars, latencies, "emulator", dirName)
                self._writePFs(gen + 1, emHof, "emulator", dirName)

                qualifiedIndividuals = [
                    ind
                    for ind in emHof
                    if ind.fitness.values[0] >= minAR
                    and ind.fitness.values[1] <= maxSecondObjective
                ]

                emMinAR = min(ars)
                emMaxLatency = max(latencies)

                TUI.appendToSolverLog(
                    f"Generation {gen}: Min AR: {emMinAR}, Max Latency: {emMaxLatency}"
                )

        return pop, qualifiedIndividuals

    def hybridSolve(
        self,
        topology: Topology,
        fgrs: "list[SFCRequest]",
        sendEGs: "Callable[[list[EmbeddingGraph]], None]",
        deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
        trafficDesign: "list[TrafficDesign]",
        trafficGenerator: TrafficGenerator,
        telemetry: Telemetry,
        popSize: int,
        experiment: str,
        type=LATENCY,
        retainPopulation: bool = False,
        linesToWrite: list[str] = [],
    ) -> None:
        """
        Run the Genetic Algorithm + Dijkstra Algorithm.

        Parameters:
            topology (Topology): the topology.
            resourceDemands (dict[str, ResourceDemand]): the resource demands.
            fgrs (list[EmbeddingGraph]): the FG Requests.
            sendEGs (Callable[[list[EmbeddingGraph]], None]): the function to send the Embedding Graphs.
            trafficDesign (list[TrafficDesign]): the traffic design.
            trafficGenerator (TrafficGenerator): the traffic generator.
            telemetry (Telemetry): telemetry instance.
            popSize (int): the population size.
            experiment (str): the experiment name.
            type (str): the optimisation objective type.
            retainPopulation (bool): whether to retain the population for the next run.

        Returns:
            None
        """

        TUI.appendToSolverLog(
            f"Running the hybrid online-offline solver for experiment: {experiment}"
        )

        expStartTime: float = timeit.default_timer()
        SCORES_DIR: str = "scores"

        expDir: str = os.path.join(self._artifactsDir, experiment)

        if not os.path.exists(expDir):
            os.makedirs(expDir)

        if not os.path.exists(os.path.join(expDir, SCORES_DIR)):
            os.makedirs(os.path.join(expDir, SCORES_DIR))

        with open(
            os.path.join(expDir, "data.csv"),
            "w",
            encoding="utf8",
        ) as topologyFile:
            topologyFile.write(
                f"method, generation, average_ar, max_ar, min_ar, average_{type}, max_{type}, min_{type}\n"
            )

        with open(
            os.path.join(expDir, "pfs.csv"),
            "w",
            encoding="utf8",
        ) as pf:
            pf.write(f"method, generation, {type}, ar\n")


        self._toolbox.register(
            "individual", self._generateRandomIndividual, self._individualContainer, topology, fgrs
        )
        self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
        )
        self._toolbox.register("mate", self._crossover)
        self._toolbox.register("mutate", self._mutate, indpb=self._indPb)
        self._toolbox.register("select", tools.selNSGA2)

        timeTaken: float = 0.0
        startTimeLimit: float = timeit.default_timer()
        pop: "list[Individual]" = (
            self._toolbox.population(n=popSize)
            if not retainPopulation or len(HybridEvolution._population) == 0
            else deepcopy(HybridEvolution._population)
        )
        TUI.appendToSolverLog(
            f"Initial population of {popSize} individuals created. Starting evolution."
        )
        gen: int = 1
        hof: tools.ParetoFront = tools.ParetoFront()
        pop, qualifiedIndividuals = self._performGeneticOperation(
            [],
            pop,
            topology,
            fgrs,
            trafficDesign,
            expDir,
            SCORES_DIR,
            gen,
            NGEN,
            MAX_MEMORY_DEMAND,
            MIN_AR,
            MAX_LATENCY if type == LATENCY else MAX_POWER,
            MIN_QUAL_IND,
            popSize,
            trafficGenerator,
            telemetry,
            sendEGs,
            deleteEGs,
            hof,
            type,
        )
        HybridEvolution._population = deepcopy(pop)

        gen = gen + 1
        timeTaken = timeTaken + ((timeit.default_timer() - startTimeLimit)/(60 * 60))
        while len(qualifiedIndividuals) < MIN_QUAL_IND and gen <= NGEN and timeTaken < TIME_LIMIT:
            startTimeLimit: float = timeit.default_timer()
            TUI.appendToSolverLog(
                f"Qualified individuals less than {MIN_QUAL_IND}. Continuing evolution."
            )
            TUI.appendToSolverLog(
                f"Generation {gen} started. Performing genetic operations."
            )
            offspring: "list[Individual]" = self._generateOffspring(pop)
            TUI.appendToSolverLog(
                f"Offspring generated for generation {gen}. Evaluating offspring."
            )
            pop, qualifiedIndividuals = self._performGeneticOperation(
                pop,
                offspring,
                topology,
                fgrs,
                trafficDesign,
                expDir,
                SCORES_DIR,
                gen,
                NGEN,
                MAX_MEMORY_DEMAND,
                MIN_AR,
                MAX_LATENCY if type == LATENCY else MAX_POWER,
                MIN_QUAL_IND,
                popSize,
                trafficGenerator,
                telemetry,
                sendEGs,
                deleteEGs,
                hof,
                type,
            )
            gen = gen + 1
            HybridEvolution._population = deepcopy(pop)
            timeTaken = timeTaken + ((timeit.default_timer() - startTimeLimit)/(60 * 60))

        expEndTime: float = timeit.default_timer()
        TUI.appendToSolverLog(f"Time taken: {expEndTime - expStartTime:.2f}s")

        names: list[str] = experiment.split("_")
        with open(
            os.path.join(expDir, "experiment.txt"),
            "w",
            encoding="utf8",
        ) as expFile:
            expFile.write(f"Experiment Name: {experiment}\n")
            expFile.write(f"Completed Date: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n")
            expFile.write(f"No. of SFCRs: {4 * int(names[0])}\n")
            expFile.write(f"Traffic Scale: {float(names[1]) * 10}\n")
            expFile.write(
                f"Traffic Pattern: {'Pattern B' if names[2] == 'True' else 'Pattern A'}\n"
            )
            expFile.write(f"Link Bandwidth: {names[3]}\n")
            expFile.write(f"No. of CPUs: {names[4]}\n")
            expFile.write(f"Time taken: {expEndTime - expStartTime:.2f}\n")
            expFile.write(f"Qualified Individuals: {len(qualifiedIndividuals)}\n")
            expFile.write(f"Minimum Acceptance Rate: {MIN_AR}\n")
            expFile.write(f"Maximum Latency: {MAX_LATENCY}\n")
            expFile.write(f"Minimum Qualified Individuals: {MIN_QUAL_IND}\n")
            expFile.write(f"Population Size: {popSize}\n")
            expFile.write(f"Maximum Number of Generations: {NGEN}\n")
            expFile.write(f"Maximum Memory Demand: {MAX_MEMORY_DEMAND}\n")
            expFile.write(f"Mutation Probability: {self._mutPb}\n")
            expFile.write(f"Gene Mutation Probability: {self._indPb}\n")
            expFile.write(f"Crossover Probability: {self._cxpPb}\n")
            expFile.write(f"Evaluation Type: {'Hybrid' if self._evaluateOnline else 'Offline'}\n")

            for line in linesToWrite:
                expFile.write(f"{line}\n")


        self._toolbox.unregister("individual")
        self._toolbox.unregister("population")
        self._toolbox.unregister("mate")
        self._toolbox.unregister("mutate")
        self._toolbox.unregister("select")
