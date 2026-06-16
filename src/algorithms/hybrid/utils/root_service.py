"""
This defines the API service for the root node of the HiGENESIS algorithm.
"""

from algorithms.hybrid.constants.root_evolver import ROOT_PATH, SELECT_NEIGHBOUR_PATH, SERVICE_PORT
from flask import Flask, jsonify, request
from algorithms.hybrid.utils.root_evolver import RootEvolver

POP_SIZE: int = 100

app: Flask = Flask(__name__)
rootEvolver: RootEvolver = RootEvolver(POP_SIZE)
root: int = rootEvolver.generateRandomRoot()
rootEvolver.setRootIndividual(root)
rootEvolver.addRootToExploredRoot(root)


@app.route(ROOT_PATH, methods=["GET"])
def getRoot() -> tuple:
    """
    Returns the current root individual.

    Returns:
        int: The current root individual.
    """

    return jsonify(RootEvolver.getRootIndividual()), 200

@app.route(SELECT_NEIGHBOUR_PATH, methods=["POST"])
def setRootNeighbour() -> tuple:
    """
    Sets the neighbour of the current root individual.

    Request JSON body:
        root (int): Root individual.
        radius (float): Radius for neighbour selection.
    """

    payload = request.get_json(silent=True) or {}

    if "root" not in payload or "radius" not in payload:
        return jsonify({"error": "Fields 'root' and 'radius' are required."}), 400

    try:
        root = int(payload["root"])
        radius = float(payload["radius"])
    except (TypeError, ValueError):
        return jsonify({"error": "Fields 'root' and 'radius' must be numeric."}), 400

    nextRoot: int = rootEvolver.selectRootNeighbour(root, radius)
    RootEvolver.setRootIndividual(nextRoot)
    rootEvolver.addRootToExploredRoot(nextRoot)
    return "", 204

def runRootService() -> None:
    """
    Starts the Flask application for the root evolver service.
    """

    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)

if __name__ == "__main__":
    runRootService()
