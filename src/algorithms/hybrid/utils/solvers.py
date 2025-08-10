"""
Defines utils used by solvers.
"""

import numpy as np


def activationFunction(x: np.ndarray) -> np.ndarray:
    """
    Applies the activation function to the input.

    Parameters:
        x (np.ndarray): the input array.

    Returns:
        np.ndarray: the output array after applying the activation function.
    """
    return np.sin(x)
