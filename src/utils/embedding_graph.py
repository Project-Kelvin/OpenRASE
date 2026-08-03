"""
Defines utils related to the embedding graph.
"""

from typing import Callable
from shared.constants.embedding_graph import TERMINAL
from shared.models.embedding_graph import VNF


def traverseVNF(vnfs: VNF, callback: Callable[[VNF, int], None], *args, shouldParseTerminal: bool = True) -> None:
    """
    Traverse the VNFs.

    Parameters:
        vnfs (VNF): The VNFs.
        callback (Callable[[VNF, int], None]): The callback function. The first two parameters of the
            callback function are VNF and depth. The depth gives you the depth of the VNF in the tree.
        *args: The arguments to be passed to the callback function.
        shouldParseTerminal (bool): Whether to parse the terminal node or not.
    """

    depth: int = 1

    def parseVNF(vnfs: VNF, depth: int = 1) -> None:
        shouldContinue: bool = True

        while shouldContinue:
            if not shouldParseTerminal and vnfs["next"] == TERMINAL:
                break

            callback(vnfs, depth, *args)

            if isinstance(vnfs['next'], list):
                depth += 1
                for nextVnf in vnfs['next']:
                    parseVNF(nextVnf, depth)

                shouldContinue = False
            else:
                vnfs = vnfs['next']

            if vnfs == TERMINAL:
                shouldContinue = False

    parseVNF(vnfs, depth)
