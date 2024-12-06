"""
This is used to calibrate the VNFs.
"""

import json
import os
from shared.utils.config import getConfig
from calibrate.calibrate import Calibrate
import click

from models.calibrate import ResourceDemand

@click.command()
@click.option("--algorithm", default="", help="The algorithm to calibrate for.")
@click.option("--vnf", default="", help="The VNF to calibrate.")
@click.option("--metric", default = "", help="The metric to calibrate.")
@click.option("--train", default = False, is_flag=True, help="If set, only ML training would be performed on existing data.")
@click.option("--epochs", help="The number of epochs to train the model.")
@click.option("--headless", default=False, is_flag=True, help="If set, the emulator would run in headless mode.")
def run(algorithm: str, vnf: str, metric: str, train: bool, epochs: int, headless: bool) -> None:
    """
    Run the calibration.

    Parameters:
        algorithm (str): The algorithm to calibrate for.
        vnf (str): The VNF to calibarte for.
        metric (str): The metric to calibrate.
        train (bool): Specifies if only training should be carried out.
        epochs (int): The number of epochs to train the model.
        headless (bool): Whether to run the emulator in headless mode.
    """

    calibrate = Calibrate()
    designFile: str = f"{getConfig()['repoAbsolutePath']}/src/calibrate/traffic-design.json"
    if algorithm == "dijkstra":
        designFile = f"{getConfig()['repoAbsolutePath']}/src/runs/simple_dijkstra_algorithm/configs/traffic-design.json"

    calibrate.calibrateVNFs(designFile, vnf, metric, headless, train, epochs)

    with open(designFile, "r", encoding="utf8") as traffic:
        design = json.load(traffic)
    maxTarget: int = max(design, key=lambda x: x["target"])["target"]
    resourceDemands: "dict[str, ResourceDemand]" = calibrate.getResourceDemands(maxTarget)
    demandsDirectory = f"{getConfig()['repoAbsolutePath']}/calibrations"
    if not os.path.exists(demandsDirectory):
        os.makedirs(demandsDirectory)
    with open(f"{demandsDirectory}/resource_demands_of_vnfs.json", "w", encoding="utf8") as demandsFile:
        json.dump(eval(str(resourceDemands)), demandsFile, indent=4)
