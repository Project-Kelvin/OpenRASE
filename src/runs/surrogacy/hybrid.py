"""
The defines teh script to run the hybrid online-offline algorithm.
"""

import json
import os
import random
from time import sleep
import click
import numpy as np
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
import tensorflow as tf
from algorithms.surrogacy.hybrid_online_offline import hybridSolver
from runs.debug import SFCSolver
from sfc.fg_request_generator import FGRequestGenerator
from sfc.sfc_emulator import SFCEmulator
from utils.topology import generateFatTreeTopology
from utils.traffic_design import generateTrafficDesignFromFile
from utils.tui import TUI

os.environ["PYTHONHASHSEED"] = "100"

# Setting the seed for numpy-generated random numbers
np.random.seed(100)

# Setting the seed for python random numbers
random.seed(100)

# Setting the graph-level random seed.
tf.random.set_seed(100)

class FGGen(FGRequestGenerator):
    """
    Class to generate FG Requests.
    """

    def __init__(self, fgs: "list[EmbeddingGraph]") -> None:
        """
        Initialize the FGGen class.
        """
        super().__init__(fgs)
        with open(
            os.path.join(
                getConfig()["repoAbsolutePath"],
                "src",
                "runs",
                "surrogacy",
                "configs",
                "forwarding-graphs.json",
            ),
            "r",
            encoding="utf8",
        ) as f:
            self.fgs = json.load(f)

    def generateRequests(self) -> "list[EmbeddingGraph]":
        """
        Generate the FG Requests.
        """

        fgsToSend: "list[EmbeddingGraph]" = []

        for i, fg in enumerate(self.fgs):
            for c in range(8):
                fgToSend: EmbeddingGraph = fg.copy()
                fgToSend["sfcrID"] = f"sfcr{i}-{c}"
                fgsToSend.append(fgToSend)

        self._orchestrator.sendRequests(fgsToSend)


trafficDesign: "list[TrafficDesign]" = [
    generateTrafficDesignFromFile(
        os.path.join(
            f"{getConfig()['repoAbsolutePath']}",
            "src",
            "runs",
            "surrogacy",
            "data",
            "requests.csv",
        ),
        0.1,
        4,
        False,
    )
]

topology: Topology = generateFatTreeTopology(4, 10, 1.0, 5120)


class HybridSolver(SFCSolver):
    """
    Class to run the hybrid online-offline algorithm.
    """

    def generateEmbeddingGraphs(self):
        """
        Generate the embedding graphs.
        """

        try:
            while self._requests.empty():
                pass
            requests: "list[EmbeddingGraph]" = []
            while not self._requests.empty():
                requests.append(self._requests.get())
                sleep(0.1)

            hybridSolver(
                topology,
                requests,
                self._orchestrator.sendEmbeddingGraphs,
                self._orchestrator.deleteEmbeddingGraphs,
                trafficDesign,
                self._trafficGenerator,
            )

        except Exception as e:
            TUI.appendToSolverLog(str(e), True)

        TUI.appendToSolverLog("Finished experiment.")


@click.command()
@click.option("--headless", is_flag=True, default=False, help="Run in headless mode.")
def run(headless: bool) -> None:
    """
    Run the hybrid online-offline algorithm.

    Parameters:
        headless (bool): Whether to run the emulator in headless mode.

    Returns:
        None
    """

    sfcEm: SFCEmulator = SFCEmulator(FGGen, HybridSolver, headless)
    sfcEm.startTest(
        topology,
        trafficDesign,
    )
    sfcEm.end()
