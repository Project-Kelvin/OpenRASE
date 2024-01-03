"""
Defines the SFCEmulator class.
"""

from typing import Type
from shared.models.topology import Topology
from mano.mano_class import MANO
from sfc.evolver import Evolver
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.traffic_generator import TrafficGenerator


class SFCEmulator():
    """
    Class that emulates SFC and allows users to run experiments.
    """

    mano: MANO = None
    sfcRequestGenerator: SFCRequestGenerator = None
    trafficGenerator: TrafficGenerator = None

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self.mano = MANO()
        self.sfcRequestGenerator = SFCRequestGenerator(self.mano.getOrchestrator())
        self.trafficGenerator = TrafficGenerator()

    def startTest(self, topology: Topology, trafficDesign, sfcRequestDesign, evolver: Type[Evolver]) -> None:
        """
        Start a test.

        Parameters:
            topology (Topology): The topology of the network.
            trafficDesign (dict): The design of the traffic generator.
            sfcRequestDesign (dict): The design of the SFC request generator.
            evolver (Evolver): The evolver.
        """

        self.mano.getInfraManager().installTopology(topology)
        self.sfcRequestGenerator.setDesign(sfcRequestDesign)
        self.trafficGenerator.setDesign(trafficDesign)

        evolver(self.mano.getOrchestrator(), self.trafficGenerator)

        self.sfcRequestGenerator.generateRequests()
