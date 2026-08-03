"""
Defines the constants used in the calibration process.
"""

import os
from shared.utils.config import getConfig


CALIBRATION_DIR: str = os.path.join(getConfig()['repoAbsolutePath'], "artifacts", "calibrations")
MODEL_NAME: str = "model.keras"
METRICS: "list[str]" = ["cpu", "memory"]
