"""
This runs the Simple Dijkstra algorithm in simulation.
"""

import copy
import json
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.utils.config import getConfig
from algorithms.simple_dijkstra_algorithm import SimpleDijkstraAlgorithm
from models.calibrate import ResourceDemand
from utils.topology import generateFatTreeTopology

config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"

topology: Topology = generateFatTreeTopology(4, 1000, 1, 512)

def run():
    """
    Simulate the algorithm.
    """

    requests = []
    resourceDemands: "dict[str, ResourceDemand]" = {
            "waf": ResourceDemand(cpu=1, memory=512, ior=0.9),
            "lb": ResourceDemand(cpu=1, memory=512, ior=0.9),
            "tm": ResourceDemand(cpu=1, memory=512, ior=0.9),
            "ha": ResourceDemand(cpu=1, memory=512, ior=0.9)
        }
    with open(f"{configPath}/forwarding-graphs.json", "r", encoding="utf8") as fgFile:
        fgs: "list[EmbeddingGraph]" = json.load(fgFile)
        for i, fg in enumerate(fgs):
            for _i in range (2):
                fg["sfcID"] = f"sfc{i}{_i}"
                requests.append(copy.deepcopy(fg))

    sda = SimpleDijkstraAlgorithm(requests, topology, resourceDemands)
    fgs, failedFGs, nodeResources = sda.run()
    print(nodeResources)
    print(f"Acceptance Ratio: {len(fgs) / (len(fgs) + len(failedFGs)) * 100:.2f}%")
