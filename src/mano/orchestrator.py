"""
Defines the class that corresponds to the Orchestrator class in the NFV architecture.
"""

from shared.models.embedding_graph import EmbeddingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from mano.infra_manager import InfraManager
from mano.telemetry import Telemetry
from mano.vnf_manager import VNFManager
from mano.sdn_controller import SDNController
from sfc.solver import Solver


class Orchestrator():
    """
    Class that corresponds to the Orchestrator class in the NFV architecture.
    """

    _infraManager: InfraManager = None
    _vnfManager: VNFManager = None
    _sdnController: SDNController = None
    _solver: Solver = None

    def __init__(self,
                 infraManager: InfraManager,
                 vnfManager: VNFManager,
                 sdnController: SDNController) -> None:
        self._infraManager = infraManager
        self._vnfManager = vnfManager
        self._sdnController = sdnController


    def sendEmbeddingGraphs(self, egs: "list[EmbeddingGraph]") -> None:
        """
        Send the embedding graphs to the orchestrator.

        Parameters:
            egs (list[EmbeddingGraph]): The list of embedding graphs.
        """

        self._vnfManager.deployEmbeddingGraphs(egs)

    def getTopology(self) -> Topology:
        """
        Get the topology from the infrastructure manager.
        """

        self._infraManager.getTopology()

    def getTelemetry(self) -> Telemetry:
        """
        Get the telemetry from the infrastructure manager.
        """

        return self._infraManager.getTelemetry()

    def installTopology(self, topology: Topology) -> None:
        """
        Install the topology using the infrastructure manager.

        Parameters:
            topology (Topology): The topology to install.
        """

        self._infraManager.installTopology(topology)

    def injectSolver(self, solver: Solver) -> None:
        """
        Inject the solver into the orchestrator.

        Parameters:
            solver (Solver): The solver to inject.
        """

        self._solver = solver

    def sendSFCRequests(self, sfcRequests: "list[SFCRequest]") -> None:
        """
        Send the SFC requests to the orchestrator.

        Parameters:
            sfcRequests (list[SFCRequest]): The list of SFC requests.
        """

        self._solver.sendSFCRequests(sfcRequests)

    def end(self) -> None:
        """
        End the orchestrator.
        """

        self._infraManager.stopNetwork()

    def startCLI(self) -> None:
        """
        Start the CLI.
        """

        self._infraManager.startCLI()
