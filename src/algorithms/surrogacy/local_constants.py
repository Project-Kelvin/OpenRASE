"""
Defines the constants used by the Surrogacy algorithm.
"""

from shared.utils.config import getConfig


BRANCH: str = "BRANCH"
SURROGACY_PATH: str = f"{getConfig()['repoAbsolutePath']}/artifacts/experiments/surrogacy"
SURROGATE_PATH: str = f"{SURROGACY_PATH}/surrogate"
SURROGATE_DATA_PATH: str = f"{SURROGATE_PATH}/data"
SURROGATE_MODELS_PATH: str = f"{SURROGATE_PATH}/models"
