"""
This is used to calibrate the VNFs.
"""

from shared.utils.config import getConfig
from calibrate.calibrate import Calibrate
import click

@click.command()
@click.option("--algorithm", default="", help="The algorithm to calibrate for.")
@click.option("--vnf", default="", help="The VNF to calibrate.")
@click.option("--metric", default = "", help="The metric to calibrate.")
@click.option("--train", default = False, is_flag=True, help="If set, only ML training would be performed on existing data.")
def run(algorithm: str, vnf: str, metric: str, train: bool) -> None:
    """
    Run the calibration.

    Parameters:
        algorithm (str): The algorithm to calibrate for.
        vnf (str): The VNF to calibarte for.
        metric (str): The metric to calibrate.
        train (bool): Specifies if only training should be carried out.
    """


    calibrate = Calibrate()
    designFile: str = ""
    if algorithm == "dijkstra":
        designFile = f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs/traffic-design.json"

    calibrate.calibrateVNFs(designFile, vnf, metric, train)
