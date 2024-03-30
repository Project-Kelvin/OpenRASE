"""
Defines the SFCEmulator class.
"""

from threading import Thread
from typing import Any, Type, Union
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.notification import TOPOLOGY_INSTALLED
from mano.mano_class import MANO
from mano.notification_system import NotificationSystem, Subscriber
from sfc.fg_request_generator import FGRequestGenerator
from sfc.solver import Solver
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.traffic_generator import TrafficGenerator


class SFCEmulator(Subscriber):
    """
    Class that emulates SFC and allows users to run experiments.
    """

    _mano: MANO = None
    _requestGenerator: Union[SFCRequestGenerator, FGRequestGenerator] = None
    _trafficGenerator: TrafficGenerator = None
    _solver: Solver = None
    _threads: "list[Thread]" = []

    def __init__(self, requestGenerator: Union[Type[SFCRequestGenerator], Type[FGRequestGenerator]], solver: Type[Solver]) -> None:
        """
        Constructor for the class.

        Parameters:
            requestGenerator (SFCRequestGenerator | FGRequestGenerator): The SFC request generator.
            solver (Type[Solver]): A child class of Solver.
        """

        self._mano = MANO()
        self._trafficGenerator = TrafficGenerator()
        self._solver = solver(self._mano.getOrchestrator(),
                              self._trafficGenerator)
        self._mano.getOrchestrator().injectSolver(self._solver)
        self._requestGenerator = requestGenerator(
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
            sfcrThread: Thread = Thread(target=self._requestGenerator.generateRequests)
            solverThread: Thread = Thread(target=self._solver.generateEmbeddingGraphs)
            sfcrThread.start()
            solverThread.start()
            self._threads.append(sfcrThread)
            self._threads.append(solverThread)

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

    def wait(self) -> None:
        """
        Wait for all threads to finish.
        """

        for thread in self._threads:
            thread.join()
