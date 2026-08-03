"""
Extracts the emulator config from the `config.yaml` file
and converts it into a Python dictionary.
"""

import yaml
from shared.models.config import Config

def getConfig() -> Config:
    """
    Extract the emulator config from the `config.yaml` file
    and convert it into a Python dictionary.

    Returns:
        Config: The emulator config.
    """

    with open("config.yaml", "r", encoding="utf-8") as file:
        config: Config = yaml.safe_load(file)

    return config
