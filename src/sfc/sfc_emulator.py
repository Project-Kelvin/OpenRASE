"""
Defines the SFCEmulator class.
"""

from typing import Any, Type
from shared.models.topology import Topology
from constants.notification import SFF_DEPLOYED
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
            sfcRequestGenerator (SFCRequestGenerator): The SFC request generator.
            solver (Type[Solver]): A child class of Solver.
        """

        self._mano = MANO()
        self._trafficGenerator = TrafficGenerator()
        self._solver = solver(self._mano.getOrchestrator(), self._trafficGenerator)
        self._sfcRequestGenerator = sfcRequestGenerator(self._solver)
        NotificationSystem.subscribe(SFF_DEPLOYED, self)


    def startTest(self, topology: Topology, trafficDesign) -> None:
        """
        Start a test.

        Parameters:
            topology (Topology): The topology of the network.
            trafficDesign (dict): The design of the traffic generator..
        """

        self._mano.getInfraManager().installTopology(topology)
        self._trafficGenerator.setDesign(trafficDesign)


    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == SFF_DEPLOYED:
            self._sfcRequestGenerator.generateRequests()
            self._solver.generateForwardingGraphs()
