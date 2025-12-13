"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is used for VNf Embedding and Dijkstra is used for link embedding.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import timeit
from typing import Callable, Tuple
from uuid import UUID, uuid4
from deap import base, creator, tools
import numpy as np
from shared.models.traffic_design import TrafficDesign
from shared.models.topology import Topology
from shared.models.embedding_graph import EmbeddingGraph
from shared.utils.config import getConfig
from algorithms.models.embedding import DecodedIndividual
from algorithms.hybrid.utils.hybrid_evaluation import HybridEvaluation
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI


class Individual(list):
    """
    Individual class for DEAP.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id: UUID = uuid4()
        self.fitness: base.Fitness = creator.MaxARMinLatency()


DecodePop = Callable[
    [list[Individual], Topology, list[EmbeddingGraph]], list[DecodedIndividual]
]
GenerateRandomIndividual = Callable[[Topology, list[EmbeddingGraph]], Individual]
Crossover = Callable[
    [Individual, Individual],
    Tuple[Individual, Individual],
]
Mutate = Callable[[Individual, float], Individual]


class HybridEvolution:
    """
    This class handles the hybrid evolution process, which includes both offline and online phases.
    It uses a surrogate model for the offline phase and an emulator for the online phase.
    """

    def __init__(
        self,
        experimentName: str,
        decodePop: DecodePop,
        generateRandomIndividual: GenerateRandomIndividual,
        crossover: Crossover,
        mutate: Mutate,
    ):
        self._decodePop: DecodePop = decodePop
        self._generateRandomIndividual: GenerateRandomIndividual = (
            generateRandomIndividual
        )
        self._crossover: Crossover = crossover
        self._mutate: Mutate = mutate
        self._toolbox: base.Toolbox = base.Toolbox()
        self._artifactsDir: str = os.path.join(
            getConfig()["repoAbsolutePath"], "artifacts", "experiments", experimentName
        )

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
        pop: "list[list[list[int]]]",
        CXPB: float,
        MUTPB: float,
    ) -> "list[list[list[int]]]":
        """
        Generate offspring from the population.

        Parameters:
            pop (list[list[list[int]]]): the population.
            CXPB (float): the crossover probability.
            MUTPB (float): the mutation probability.

        Returns:
            offspring (list[list[list[int]]]): the offspring.
        """

        offspring: "list[list[list[int]]]" = list(map(self._toolbox.clone, pop))
        random.shuffle(offspring)

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:

                self._toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values
                child1.id = uuid4()
                child2.id = uuid4()

        for mutant in offspring:
            if random.random() < MUTPB:
                self._toolbox.mutate(mutant)

                del mutant.fitness.values

        return offspring

    def _performGeneticOperation(
        self,
        parent: list[Individual],
        pop: list[Individual],
        topology: Topology,
        fgrs: list[EmbeddingGraph],
        trafficDesign: list[TrafficDesign],
        dirName: str,
        scoresDir: str,
        gen: int,
        ngen: int,
        maxMemoryDemand: float,
        minAR: float,
        maxLatency: float,
        minQualifiedInds: int,
        popSize: int,
        trafficGenerator: TrafficGenerator,
        sendEGs: "Callable[[list[EmbeddingGraph]], None]",
        deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
        hof: tools.ParetoFront,
    ) -> "tuple[list[Individual], list[Individual]]":
        """
        Perform the genetic operation.

        Parameters:
            parent (list[Individual]): the parent population.
            pop (list[Individual]): the current population.
            topology (Topology): the topology.
            fgrs (list[EmbeddingGraph]): the Forwarding Graph Requests.
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
            sendEGs (Callable[[list[EmbeddingGraph]], None]): function to send Embedding Graphs.
            deleteEGs (Callable[[list[EmbeddingGraph]], None]): function to delete Embedding Graphs.
            hof (tools.ParetoFront): hall of fame for storing best individuals.

        Returns:
            tuple[list[Individual], list[Individual]: the updated population and qualified individuals.
        """

        populationEG: "list[DecodedIndividual]" = self._decodePop(pop, topology, fgrs)
        HybridEvaluation.cacheForOffline(
            populationEG, trafficDesign, topology, gen, isAvgOnly=True
        )
        HybridEvaluation.saveCachedLatency(
            os.path.join(dirName, scoresDir, f"gen_{gen}.csv")
        )
        startTime: int = timeit.default_timer()
        with ProcessPoolExecutor() as executor:
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
        endTime: int = timeit.default_timer()
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

        qualifiedIndividuals = [
            ind
            for ind in hof
            if ind.fitness.values[0] >= minAR and ind.fitness.values[1] <= maxLatency
        ]

        TUI.appendToSolverLog(
            f"Qualified Individuals: {len(qualifiedIndividuals)}/{minQualifiedInds}"
        )

        if len(qualifiedIndividuals) >= minQualifiedInds:
            TUI.appendToSolverLog(
                f"Finished the evolution of weights using surrogate at generation {gen}."
            )
            TUI.appendToSolverLog(
                f"Number of qualified individuals: {len(qualifiedIndividuals)}"
            )

            # ---------------------------------------------------------------------------------------------
            # Start the online phase of the hybrid evolution
            # ---------------------------------------------------------------------------------------------

            for ind in qualifiedIndividuals:
                del ind.fitness.values

            emHof = tools.ParetoFront()

            populationEG: "list[DecodedIndividual]" = self._decodePop(
                qualifiedIndividuals, topology, fgrs
            )
            HybridEvaluation.cacheForOnline(populationEG, trafficDesign)
            for ind in populationEG:
                ar, latency = HybridEvaluation.evaluationOnEmulator(
                    ind,
                    fgrs,
                    gen,
                    ngen,
                    sendEGs,
                    deleteEGs,
                    trafficDesign,
                    trafficGenerator,
                    topology,
                    maxMemoryDemand,
                )
                qualifiedIndividuals[ind[0]].fitness.values = (ar, latency)

                for p in pop:
                    if p.id == qualifiedIndividuals[ind[0]].id:
                        p.fitness.values = (ar, latency)
                        break

            emHof.update(qualifiedIndividuals)

            ars = [ind.fitness.values[0] for ind in qualifiedIndividuals]
            latencies = [ind.fitness.values[1] for ind in qualifiedIndividuals]

            self._writeData(gen + 0.1, ars, latencies, "emulator", dirName)
            self._writePFs(gen + 0.1, emHof, "emulator", dirName)

            qualifiedIndividuals = [
                ind
                for ind in emHof
                if ind.fitness.values[0] >= minAR
                and ind.fitness.values[1] <= maxLatency
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
        fgrs: "list[EmbeddingGraph]",
        sendEGs: "Callable[[list[EmbeddingGraph]], None]",
        deleteEGs: "Callable[[list[EmbeddingGraph]], None]",
        trafficDesign: "list[TrafficDesign]",
        trafficGenerator: TrafficGenerator,
        popSize: int,
        experiment: str,
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
            popSize (int): the population size.
            experiment (str): the experiment name.

        Returns:
            None
        """

        TUI.appendToSolverLog(
            f"Running the hybrid online-offline solver for experiment: {experiment}"
        )

        expStartTime: int = timeit.default_timer()
        NGEN: int = 500
        MAX_MEMORY_DEMAND: int = 2
        MAX_LATENCY: int = 100
        MIN_AR: float = 1.0
        MIN_QUAL_IND: int = 1
        CXPB: float = 1.0
        INDPB: float = 0.2
        MUTPB: float = 0.8
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
                "method, generation, average_ar, max_ar, min_ar, average_latency, max_latency, min_latency\n"
            )

        with open(
            os.path.join(expDir, "pfs.csv"),
            "w",
            encoding="utf8",
        ) as pf:
            pf.write("method, generation, latency, ar\n")

        creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))

        self._toolbox.register(
            "individual", self._generateRandomIndividual, Individual, topology, fgrs
        )
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("mate", self._crossover)
        self._toolbox.register("mutate", self._mutate, indpb=INDPB)
        self._toolbox.register("select", tools.selNSGA2)

        pop: "list[Individual]" = self._toolbox.population(n=popSize)

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
            MAX_LATENCY,
            MIN_QUAL_IND,
            popSize,
            trafficGenerator,
            sendEGs,
            deleteEGs,
            hof,
        )

        gen = gen + 1

        while len(qualifiedIndividuals) < MIN_QUAL_IND and gen <= NGEN:
            offspring: "list[Individual]" = self._generateOffspring(
                pop, CXPB, MUTPB
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
                MAX_LATENCY,
                MIN_QUAL_IND,
                popSize,
                trafficGenerator,
                sendEGs,
                deleteEGs,
                hof,
            )
            gen = gen + 1

        expEndTime: int = timeit.default_timer()
        TUI.appendToSolverLog(f"Time taken: {expEndTime - expStartTime:.2f}s")

        names: list[str] = experiment.split("_")
        with open(
            os.path.join(expDir, "experiment.txt"),
            "w",
            encoding="utf8",
        ) as expFile:
            expFile.write(f"No. of SFCRs: {4 * int(names[0])}\n")
            expFile.write(f"Traffic Scale: {float(names[1]) * 10}\n")
            expFile.write(
                f"Traffic Pattern: {'Pattern B' if names[2] == 'True' else 'Pattern A'}\n"
            )
            expFile.write(f"Link Bandwidth: {names[3]}\n")
            expFile.write(f"No. of CPUs: {names[4]}\n")
            expFile.write(f"Time taken: {expEndTime - expStartTime:.2f}\n")

        self._toolbox.unregister("individual")
        self._toolbox.unregister("population")
        self._toolbox.unregister("mate")
        self._toolbox.unregister("mutate")
        self._toolbox.unregister("select")
