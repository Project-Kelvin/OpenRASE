"""
Defines utils used by solvers.
"""

import numpy as np


def activationFunction(x: np.ndarray, activation: str = "sin") -> np.ndarray:
    """
    Applies the activation function to the input.

    Parameters:
        x (np.ndarray): the input array.
        activation (str): the type of activation function to apply.

    Returns:
        np.ndarray: the output array after applying the activation function.
    """
    if activation == "sin":
        return np.sin(x)
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "linear":
        return x
    else:
        raise ValueError("Unsupported activation function")
