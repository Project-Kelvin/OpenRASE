"""
This defines the API service for the root node of the HiGENESIS algorithm.
"""


from algorithms.hybrid.constants.root_evolver import GENERATE_RANDOM_ROOT_PATH, INIT, SELECT_NEIGHBOUR_PATH, SERVICE_PORT, SET_ROOT_FITNESS_PATH
from flask import Flask, jsonify, request
from algorithms.hybrid.utils.root.root_search_space import RootSearchSpace


app: Flask = Flask(__name__)
rootSearchSpace: RootSearchSpace # = RootSearchSpace(100)  # Initialize with a default population size of 100

@app.route(INIT, methods=["POST"])
def init() -> tuple:
    """
    Initializes the root search space with the given population size.

    Returns:
        None
    """

    payload = request.get_json(silent=True) or {}

    if "popSize" not in payload:
        return jsonify({"error": "Field 'popSize' is required."}), 400

    try:
        popSize: int = int(payload["popSize"])
        rootSearchSpace = RootSearchSpace(popSize)
    except (TypeError, ValueError):
        return jsonify({"error": "Field 'popSize' must be an integer."}), 400

    return "", 204

@app.route(GENERATE_RANDOM_ROOT_PATH, methods=["POST"])
def generateRandomRoot() -> tuple:
    """
    Generates a random root individual and sets it as the current root.

    Returns:
        int: The newly generated root individual.
    """

    newRoot: int = rootSearchSpace.generateRandomRoot()

    return jsonify(newRoot), 201

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

    root: int = rootSearchSpace.selectNextRoot(root, radius)

    return jsonify(root), 204

@app.route(SET_ROOT_FITNESS_PATH, methods=["POST"])
def setRootFitness() -> tuple:
    """
    Sets the fitness of the current root individual.

    Request JSON body:
        root (int): Root individual.
        fitness (float): Fitness value.
    """

    payload = request.get_json(silent=True) or {}

    if "root" not in payload or "fitness" not in payload:
        return jsonify({"error": "Fields 'root' and 'fitness' are required."}), 400

    try:
        root: int = int(payload["root"])
        fitness: float = float(payload["fitness"])
    except (TypeError, ValueError):
        return jsonify({"error": "Fields 'root' and 'fitness' must be numeric."}), 400

    rootSearchSpace.setRootFitness(root, fitness)

    return "", 204

def runRootService() -> None:
    """
    Starts the Flask application for the root evolver service.
    """

    app.run(debug=True, host="0.0.0.0", port=SERVICE_PORT)

if __name__ == "__main__":
    runRootService()
