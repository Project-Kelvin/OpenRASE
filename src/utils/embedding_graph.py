"""
Defines utils related to the forwarding graph.
"""

from typing import Callable
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF


def traverseVNF(vnfs: VNF, callback: Callable[[VNF], None], *args, shouldParseTerminal: bool = True) -> None:
    """
    Traverse the VNFs.

    Parameters:
        vnfs (VNF): The VNFs.
        callback (Callable[[VNF], None]): The callback function.
        *args: The arguments to be passed to the callback function.
        shouldParseTerminal (bool): Whether to parse the terminal node or not.
    """

    def parseVNF(vnfs: VNF) -> None:
        shouldContinue: bool = True

        while shouldContinue:
            if not shouldParseTerminal and vnfs["next"] == TERMINAL:
                break

            callback(vnfs, *args)

            if isinstance(vnfs['next'], list):
                for nextVnf in vnfs['next']:
                    parseVNF(nextVnf)

                shouldContinue = False
            else:
                vnfs = vnfs['next']

            if vnfs == TERMINAL:
                shouldContinue = False

    parseVNF(vnfs)
