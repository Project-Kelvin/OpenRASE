"""
This is used to calibrate the VNFs.
"""

from shared.utils.config import getConfig
from calibrate.calibrate import Calibrate
import click

@click.command()
@click.option("--algorithm", default="", help="The algorithm to calibrate for.")
def run(algorithm: str) -> None:
    """
    Run the calibration.

    Parameters:
        algorithm (str): The algorithm to calibrate for.
    """


    calibrate = Calibrate()
    if algorithm == "dijkstra":
        calibrate.calibrateVNFs(
            f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs/traffic-design.json")
    else:
        calibrate.calibrateVNFs()
