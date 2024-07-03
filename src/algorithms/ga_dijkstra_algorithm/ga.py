"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from deap import base, creator, tools

from algorithms.ga_dijkstra_algorithm.utils import algorithm, evalutaionThunk, generateRandomIndividual, generateRandomVNFEmbedding, mutate
from models.calibrate import ResourceDemand


NO_OF_INDIVIDUALS: int = 10

def GADijkstraAlgorithm(topology: Topology, resourceDemands: "dict[str, ResourceDemand]", fgrs: "list[EmbeddingGraph]") -> None:
    """
    Run the Genetic Algorithm + Dijkstra Algorithm.

    Parameters:
        topology (Topology): the topology.
        resourceDemands (dict[str, ResourceDemand]): the resource demands.
        fgrs (list[EmbeddingGraph]): the FG Requests.

    Returns:
        None
    """


    creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
    creator.create("Individual", list, fitness=creator.MaxARMinLatency)

    toolbox:base.Toolbox = base.Toolbox()

    toolbox.register("individual", generateRandomIndividual, creator.Individual, topology, resourceDemands, fgrs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalutaionThunk())
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", mutate, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=NO_OF_INDIVIDUALS)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 10

    gen = 0
    hof = tools.ParetoFront()
    while gen < NGEN:
        gen = gen + 1
        offspring = algorithm(pop, toolbox, CXPB, MUTPB, topology, resourceDemands, fgrs)
        pop = pop + offspring
        fits = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
        pop = toolbox.select(pop, k=NO_OF_INDIVIDUALS)
        hof.update(pop)

    hof
