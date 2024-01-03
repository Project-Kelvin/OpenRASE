"""
Defines the SFCRequestGenerator class.
"""

from shared.models.sfc_request import SFCRequest
from mano.orchestrator import Orchestrator


class SFCRequestGenerator():
    """
    Class that generates SFC requests.
    """

    orchestrator: Orchestrator = None

    def __init__(self, orchestrator: Orchestrator) -> None:
        """
        Constructor for the class.
        """

        self.orchestrator = orchestrator

    def setDesign(self, design) -> None:
        """
        Set the design of the SFC request generator.

        Parameters:
            design (dict): The design of the SFC request generator.
        """

        pass

    def generateRequests(self) -> None:
        """
        Generate SFC requests.
        """

        sfcRequests: "list[SFCRequest]" = []
        self.orchestrator.sendSFCRequests(sfcRequests)
