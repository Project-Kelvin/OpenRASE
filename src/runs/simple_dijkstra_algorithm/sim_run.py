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
from utils.topology import generateFatTreeTopology

config: Config = getConfig()
configPath: str = f"{config['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs"

topology: Topology = generateFatTreeTopology(4, 1000, 1, 512)

def run():
    """
    Simulate the algorithm.
    """

    requests = []
    trafficDesignPath = f"{configPath}/traffic-design.json"
    with open(trafficDesignPath, "r", encoding="utf8") as traffic:
        design = json.load(traffic)
    maxTarget: int = max(design, key=lambda x: x["target"])["target"]


    with open(f"{configPath}/forwarding-graphs.json", "r", encoding="utf8") as fgFile:
        fgs: "list[EmbeddingGraph]" = json.load(fgFile)
        for i, fg in enumerate(fgs):
            for _i in range (1):
                fg["sfcrID"] = f"sfc{i}{_i}"
                requests.append(copy.deepcopy(fg))
    sda = SimpleDijkstraAlgorithm(requests, topology, maxTarget)
    fgs, failedFGs, nodeResources = sda.run()
    print(nodeResources)
    print(f"Acceptance Ratio: {len(fgs) / (len(fgs) + len(failedFGs)) * 100:.2f}%")
