"""
Defines the MANO class that performs the Management and Operation functions.
"""

from mano.infra_manager import InfraManager
from mano.orchestrator import Orchestrator
from mano.vnf_manager import VNFManager
from mano.sdn_controller import SDNController


class MANO():
    """
    Class that performs Management and Operations functions.
    """


    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self._sdnController: SDNController = SDNController()
        self._infraManager: InfraManager = InfraManager(self._sdnController)
        self._vnfManager: VNFManager = VNFManager(self._infraManager)
        self._orchestrator: Orchestrator = Orchestrator(
            self._infraManager, self._vnfManager, self._sdnController)

    def getOrchestrator(self) -> Orchestrator:
        """
        Get the orchestrator.

        Returns:
            Orchestrator: The orchestrator.
        """

        return self._orchestrator
