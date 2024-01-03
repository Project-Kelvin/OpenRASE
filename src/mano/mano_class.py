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

    infraManager: InfraManager = None
    vnfManager: VNFManager = None
    sdnController: SDNController = None
    orchestrator: Orchestrator = None

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self.sdnController = SDNController()
        self.infraManager = InfraManager(self.sdnController)
        self.vnfManager = VNFManager(self.infraManager)
        self.orchestrator = Orchestrator(
            self.infraManager, self.vnfManager, self.sdnController)

    def getInfraManager(self) -> InfraManager:
        """
        Get the infrastructure manager.

        Returns:
            InfraManager: The infrastructure manager.
        """

        return self.infraManager

    def getVNFManager(self) -> VNFManager:
        """
        Get the VNF manager.

        Returns:
            VNFManager: The VNF manager.
        """

        return self.vnfManager

    def getSDNController(self) -> SDNController:
        """
        Get the SDN controller.

        Returns:
            SDNController: The SDN controller.
        """

        return self.sdnController

    def getOrchestrator(self) -> Orchestrator:
        """
        Get the orchestrator.

        Returns:
            Orchestrator: The orchestrator.
        """

        return self.orchestrator
