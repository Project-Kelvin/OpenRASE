"""
This trains teh surrogate model.
"""

from algorithms.surrogacy.surrogate import Surrogate


def run():
    """
    Runs the surrogate model.
    """

    surrogate: Surrogate = Surrogate()
    surrogate.train()
