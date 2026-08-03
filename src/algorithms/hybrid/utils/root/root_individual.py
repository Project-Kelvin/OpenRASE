"""
This defines the root individual for the HiGENESIS algorithm.
"""

class RootIndividual:
    """
    The RootIndividual class represents an individual in the root search space of the HiGENESIS algorithm.
    It provides methods to get and set the fitness of the individual.
    """

    def __init__(self):
        """
        Initializes the RootIndividual with the given root value.

        Parameters:
            root (int): The root value representing the individual.
        """

        self._root: int = -1

    def getRoot(self) -> int:
        """
        Returns the root value of the individual.
        """
        return self._root

    def setRoot(self, root: int) -> None:
        """
        Sets the root value of the individual.

        Parameters:
            root (int): The new root value to set.
        """

        self._root = root
