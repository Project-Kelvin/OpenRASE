"""
This is used to calibrate the VNFs.
"""

from calibrate.calibrate import Calibrate

def run() -> None:
    """
    Run the calibration.
    """

    calibrate = Calibrate()
    calibrate.calibrateVNFs()
