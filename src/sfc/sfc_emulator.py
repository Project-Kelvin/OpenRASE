"""
Defines the SFCEmulator class.
"""

from threading import Thread
from typing import Any, Optional, Type, Union
from time import monotonic, sleep
from shared.models.topology import Topology
from shared.models.traffic_design import TrafficDesign
from constants.notification import TOPOLOGY_INSTALLED, TOPOLOGY_INSTALL_FAILED
from mano.mano_class import MANO
from mano.notification_system import NotificationSystem, Subscriber
from sfc.fg_request_generator import FGRequestGenerator
from sfc.solver import Solver
from sfc.sfc_request_generator import SFCRequestGenerator
from sfc.traffic_generator import TrafficGenerator
from utils.tui import TUI
from docker import from_env, DockerClient

class SFCEmulator(Subscriber):
    """
    Class that emulates SFC and allows users to run experiments.
    """


    def __init__(self, requestGenerator: Union[Type[SFCRequestGenerator], Type[FGRequestGenerator]], solver: Type[Solver], headless: bool = False) -> None:
        """
        Constructor for the class.

        Parameters:
            requestGenerator (SFCRequestGenerator | FGRequestGenerator): The SFC request generator.
            solver (Type[Solver]): A child class of Solver.
            headless (bool): Whether to run the emulator in headless mode.
        """

        TUI.setMode(headless)
        self._mano: MANO = MANO()
        self._trafficGenerator: TrafficGenerator = TrafficGenerator()
        Thread(target=self._trafficGenerator.startParentContainer).start()
        self._solver: Solver = solver(self._mano.getOrchestrator(),
                              self._trafficGenerator)
        self._mano.getOrchestrator().injectSolver(self._solver)
        self._requestGenerator: Union[SFCRequestGenerator, FGRequestGenerator] = requestGenerator(
            self._mano.getOrchestrator())
        self._threads: "list[Thread]" = []
        NotificationSystem.subscribe(TOPOLOGY_INSTALLED, self)
        NotificationSystem.subscribe(TOPOLOGY_INSTALL_FAILED, self)
        self._topologyInstalled: bool = False
        self._topologyInstallFailed: bool = False
        self._topologyInstallError: Optional[str] = None

    def startTest(self, topology: Topology, trafficDesign: "list[TrafficDesign]") -> None:
        """
        Start a test.

        Parameters:
            topology (Topology): The topology of the network.
            trafficDesign (dict): The design of the traffic generator.
        """

        self._trafficGenerator.setDesign(trafficDesign)
        Thread(target=self._mano.getOrchestrator().installTopology, args=(topology,)).start()
        TUI.init()
        self._wait()

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == TOPOLOGY_INSTALLED:
            sfcrThread: Thread = Thread(target=self._requestGenerator.generateRequests)
            solverThread: Thread = Thread(target=self._solver.generateEmbeddingGraphs)
            TUI.appendToLog("Starting Request Generator.")
            sfcrThread.start()
            TUI.appendToLog("Starting Solver.")
            solverThread.start()
            self._topologyInstalled = True
            self._threads.append(sfcrThread)
            self._threads.append(solverThread)
        elif topic == TOPOLOGY_INSTALL_FAILED:
            self._topologyInstallFailed = True
            self._topologyInstallError = str(args[0]) if len(args) > 0 else "unknown error"

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
        NotificationSystem.unsubscribeAll()

        try:
            client: DockerClient = from_env()
            client.containers.prune()
            client.images.prune()
            client.networks.prune()
            client.volumes.prune()
        except Exception as e:
            TUI.appendToLog(f"Error while pruning Docker resources: {e}", True)

        TUI.exit()

    def _wait(self) -> None:
        """
        Wait for all threads to finish.
        """

        installTimeoutSec: float = 120.0
        startTime: float = monotonic()

        while not self._topologyInstalled and not self._topologyInstallFailed:
            if monotonic() - startTime > installTimeoutSec:
                raise TimeoutError(
                    "Timed out while waiting for topology installation. "
                    "Check logs for installation errors."
                )
            sleep(0.05)

        if self._topologyInstallFailed:
            raise RuntimeError(
                f"Topology installation failed: {self._topologyInstallError}"
            )

        for thread in self._threads:
            thread.join()

    def getSolver(self) -> Solver:
        """
        Get the solver.

        Returns:
            Solver: The solver.
        """

        return self._solver
