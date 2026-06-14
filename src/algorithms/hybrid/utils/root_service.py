"""
This defines the API service for the root node of the HiGENESIS algorithm.
"""

from algorithms.hybrid.constants.root_evolver import ROOT_PATH, SELECT_NEIGHBOUR_PATH
from algorithms.hybrid.models.root_evolver import SelectNeighbour
from fastapi import FastAPI
from algorithms.hybrid.utils.root_evolver import RootEvolver

POP_SIZE: int = 100

app: FastAPI = FastAPI()
rootEvolver: RootEvolver = RootEvolver(POP_SIZE)
root: int = rootEvolver.generateRandomRoot()
rootEvolver.setRootIndividual(root)
rootEvolver.addRootToExploredRoot(root)


@app.get(ROOT_PATH)
def getRoot() -> int:
    """
    Returns the current root individual.

    Returns:
        int: The current root individual.
    """

    return RootEvolver.getRootIndividual()

@app.post(SELECT_NEIGHBOUR_PATH)
def setRootNeighbour(selectNeighbour: SelectNeighbour) -> None:
    """
    Sets the neighbour of the current root individual.

    Parameters:
        selectNeighbour (SelectNeighbour): The request body containing the neighbour information.
    """

    nextRoot: int = rootEvolver.selectRootNeighbour(selectNeighbour.root, selectNeighbour.radius)
    RootEvolver.setRootIndividual(nextRoot)
    rootEvolver.addRootToExploredRoot(nextRoot)
