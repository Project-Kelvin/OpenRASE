"""
This defines a Genetic Algorithm (GA) to produce an Embedding Graph from a Forwarding Graph.
GA is sued for VNf Embedding and Dijkstra isu sed for link embedding.
"""

from deap import base, creator, tools, algorithms

from algorithms.ga_dijkstra_algorithm.utils import algorithm, evalutaionThunk, generateRandomIndividual, generateRandomVNFEmbedding, mutate


NO_OF_HOSTS: int = 5
NO_OF_VNFS: int = 5
NO_OF_INDIVIDUALS: int = 10

creator.create("MaxARMinLatency", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.MaxARMinLatency)

toolbox:base.Toolbox = base.Toolbox()

toolbox.register("individual", generateRandomIndividual, creator.Individual, NO_OF_HOSTS, NO_OF_VNFS, topo, resourceDemands, sfcr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalutaionThunk())
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", mutate, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=NO_OF_INDIVIDUALS)

CXPB, MUTPB, NGEN = 0.5, 0.2, 10

gen = 0

while gen < NGEN:
    gen = gen + 1
    fits = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fits):
        ind.fitness.values = fit
    offspring = toolbox.select(pop, k=len(pop))
    pop = algorithm(offspring, toolbox, cxpb=CXPB, mutpb=MUTPB, topo, resourceDemands, sfcr)
