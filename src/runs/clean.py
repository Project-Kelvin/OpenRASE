"""
This defines the function to clean the log files.
"""

import shutil

import click
from shared.models.config import Config
from shared.utils.config import getConfig
import os

def removeFiles(log_dir: str) -> None:
    """
    This function removes the log files.
    """

    for file_name in os.listdir(log_dir):
        if file_name != ".gitignore":
            file_path = os.path.join(log_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

@click.command()
@click.option("--logs", default=False, type=bool, is_flag=True, help="Delete the log files.")
@click.option("--docker", default=False, type=bool, is_flag=True, help="Delete the docker containers.")
def clean(logs: bool, docker: bool) -> None:
    """
    This function cleans the log files and docker containers.
    """

    def cleanLogs() -> None:
        """
        This function cleans the log files.
        """
        # Clean the log files
        print("Cleaning the log files.")
        # Code to clean the log files

        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/ids/shared/logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/ids/shared/node-logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/dpi/shared/logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/dpi/shared/node-logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/ips/shared/logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/ips/shared/node-logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/influxdb/data")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/node-sfcc/shared/node-logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/node-sff/shared/node-logs")
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/files/tm/shared/node-logs")

    def cleanDocker() -> None:
        """
        This function cleans the docker containers.
        """

        # Clean the docker containers
        print("Cleaning the docker containers.")
        # Code to clean the docker containers
        removeFiles(f"{getConfig()['repoAbsolutePath']}/docker/registry")

    if logs:
        cleanLogs()

    if docker:
        cleanDocker()

    if not logs and not docker:
        cleanLogs()
        cleanDocker()
