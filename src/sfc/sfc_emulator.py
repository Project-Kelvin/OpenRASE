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

    _mano: MANO = None
    _sfcRequestGenerator: SFCRequestGenerator = None
    _trafficGenerator: TrafficGenerator = None

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self._mano = MANO()
        self._sfcRequestGenerator = SFCRequestGenerator(self._mano.getOrchestrator())
        self._trafficGenerator = TrafficGenerator()

    def startTest(self, topology: Topology, trafficDesign, sfcRequestDesign, evolver: Type[Evolver]) -> None:
        """
        Start a test.

        Parameters:
            topology (Topology): The topology of the network.
            trafficDesign (dict): The design of the traffic generator.
            sfcRequestDesign (dict): The design of the SFC request generator.
            evolver (Evolver): The evolver.
        """

        self._mano.getInfraManager().installTopology(topology)
        self._mano.getVNFManager().deploySFF()
        self._sfcRequestGenerator.setDesign(sfcRequestDesign)
        self._trafficGenerator.setDesign(trafficDesign)

        evolver(self._mano.getOrchestrator(), self._trafficGenerator)

        self._sfcRequestGenerator.generateRequests()
