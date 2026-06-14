"""
This defines the API service for the root node of the HiGENESIS algorithm.
"""

from fastapi import FastAPI
from src.algorithms.hybrid.utils.root_evolver import RootEvolver

app: FastAPI = FastAPI()

@app.get("/root")
def get_root() -> int:
    """
    Returns the current root individual.

    Returns:
        int: The current root individual.
    """

    return RootEvolver.getRootIndividual()

@app.
