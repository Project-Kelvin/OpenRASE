"""
This file contains the code to run the InfluxDB container.
"""

import click
from docker import from_env
from docker.models.containers import Container
from shared.utils.config import getConfig

from constants.container import TAG
from sfc.traffic_generator import INFLUX_DB_CONFIG, INFLUXDB


client= from_env()

@click.command()
@click.option("--stop", is_flag=True, help="Stop the InfluxDB container.")
def run(stop: bool) -> None:
    """
    Run the InfluxDB container.

    Parameters:
        stop (bool): Stop the InfluxDB container.
    """

    if stop:
        container: Container = client.containers.get(INFLUXDB)

        if container:
            container.stop()
            print(f"InfluxDB container stopped.")
        return
    else:
        container: Container = client.containers.run(
            f"{TAG}/influxdb:latest",
            name=INFLUXDB,
            detach=True,
            environment={
                "DOCKER_INFLUXDB_INIT_MODE": "setup",
                "DOCKER_INFLUXDB_INIT_USERNAME": INFLUX_DB_CONFIG["USERNAME"],
                "DOCKER_INFLUXDB_INIT_PASSWORD": INFLUX_DB_CONFIG["PASSWORD"],
                "DOCKER_INFLUXDB_INIT_ORG": INFLUX_DB_CONFIG["ORG"],
                "DOCKER_INFLUXDB_INIT_BUCKET": INFLUX_DB_CONFIG["BUCKET"],
                "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN": INFLUX_DB_CONFIG["TOKEN"]
            },
            ports={"8086/tcp": 8086},
            volumes=[f"{getConfig()['repoAbsolutePath']}/docker/files/influxdb/data:/var/lib/influxdb2"],
            auto_remove=True
        )

        print(f"InfluxDB running on: http://localhost:8086")
