"""
Defines the SFCEmulator class.
"""

from threading import Thread
from typing import Any, Type
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.notification import TOPOLOGY_INSTALLED
from mano.mano_class import MANO
from mano.notification_system import NotificationSystem, Subscriber
from sfc.solver import Solver
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.traffic_generator import TrafficGenerator


class SFCEmulator(Subscriber):
    """
    Class that emulates SFC and allows users to run experiments.
    """

    _mano: MANO = None
    _sfcRequestGenerator: SFCRequestGenerator = None
    _trafficGenerator: TrafficGenerator = None
    _solver: Solver = None

    def __init__(self, sfcRequestGenerator: Type[SFCRequestGenerator], solver: Type[Solver]) -> None:
        """
        Constructor for the class.

        Parameters:
            sfcRequestGenerator (ISFCRequestGenerator): The SFC request generator.
            solver (Type[Solver]): A child class of Solver.
        """

        self._mano = MANO()
        self._trafficGenerator = TrafficGenerator()
        self._solver = solver(self._mano.getOrchestrator(),
                              self._trafficGenerator)
        self._mano.getOrchestrator().injectSolver(self._solver)
        self._sfcRequestGenerator = sfcRequestGenerator(
            self._mano.getOrchestrator())
        NotificationSystem.subscribe(TOPOLOGY_INSTALLED, self)

    def startTest(self, topology: Topology, trafficDesign: "list[TrafficDesign]") -> None:
        """
        Start a test.

        Parameters:
            topology (Topology): The topology of the network.
            trafficDesign (dict): The design of the traffic generator..
        """

        self._trafficGenerator.setDesign(trafficDesign)
        self._mano.getOrchestrator().installTopology(topology)

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == TOPOLOGY_INSTALLED:
            Thread(target=self._sfcRequestGenerator.generateRequests).start()
            Thread(target=self._solver.generateEmbeddingGraphs).start()

    def startCLI(self) -> None:
        """
        Start the command line interface.
        """

        self._mano.getOrchestrator().startCLI()

    def end(self) -> None:
        """
        End the emulator.
        """

        self._mano.getOrchestrator().end()
        self._trafficGenerator.end()
