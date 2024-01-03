"""
Defines the class that corresponds to the Orchestrator class in the NFV architecture.
"""

from shared.models.forwarding_graph import ForwardingGraph
from shared.models.sfc_request import SFCRequest
from shared.models.topology import Topology
from mano.infra_manager import InfraManager
from mano.vnf_manager import VNFManager
from mano.sdn_controller import SDNController


class Orchestrator():
    """
    Class that corresponds to the Orchestrator class in the NFV architecture.
    """

    sfcRequests: "list[SFCRequest]" = []
    infraManager: InfraManager = None
    vnfManager: VNFManager = None
    sdnController: SDNController = None

    def __init__(self,
                 infraManager: InfraManager,
                 vnfManager: VNFManager,
                 sdnController: SDNController) -> None:
        self.infraManager = infraManager
        self.vnfManager = vnfManager
        self.sdnController = sdnController

    def sendSFCRequests(self, sfcRequests: "list[SFCRequest]") -> None:
        """
        Send the SFC requests to the orchestrator.

        Parameters:
            sfcRequests (list[ForwardingGraph]): The list of SFC requests.
        """

        sfcRequests.append(sfcRequests)


    def sendForwardGraphs(self, fgs: "list[ForwardingGraph]") -> None:
        """
        Send the forwarding graphs to the orchestrator.

        Parameters:
            fgs (list[ForwardingGraph]): The list of forwarding graphs.
        """

        self.vnfManager.deployForwardingGraphs(fgs)

    def getTopology(self) -> Topology:
        """
        Get the topology from the infrastructure manager.
        """

        self.infraManager.getTopology()

    def getTelemetry(self) -> None:
        """
        Get the telemetry from the infrastructure manager.
        """

        self.infraManager.getTelemetry()
