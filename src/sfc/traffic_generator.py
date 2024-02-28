"""
Defines the TrafficGenerator class.
"""

import json
import random
from time import sleep
from typing import Any
from jinja2 import Template, TemplateSyntaxError
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from influxdb_client import InfluxDBClient
from constants.container import DIND_IMAGE, TAG
from constants.notification import EMBEDDING_GRAPH_DELETED, EMBEDDING_GRAPH_DEPLOYED
from constants.topology import SFCC, TRAFFIC_GENERATOR
from mano.notification_system import NotificationSystem, Subscriber
from utils.container import connectToDind, getContainerIP
from docker import DockerClient, from_env


TG_HOME_PATH: str = "/home/docker/compose"
INFLUX_DB_CONFIG = {
    "USERNAME": "admin",
    "PASSWORD": "password",
    "ORG": "SFC_Test",
    "BUCKET": "SFC_Test",
    "TOKEN": "K6_INFLUXDB"
}
INFLUXDB: str = "influxdb"
K6: str = "K6"

class TrafficGenerator(Subscriber):
    """
    Class that generates traffic.
    """

    _design: "list[TrafficDesign]" = []
    _tgClient: DockerClient = None
    _influxDBClient: InfluxDBClient = None

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        config: Config = getConfig()
        dockerClient: DockerClient = from_env()

        dockerClient.containers.run(DIND_IMAGE, detach=True, name=TRAFFIC_GENERATOR, privileged=True, volumes=[
            f"{config['repoAbsolutePath']}/docker/compose:{TG_HOME_PATH}"
        ], command="dockerd", ports={"8086/tcp": 8086})
        sleep(20)
        self._tgClient = connectToDind(TRAFFIC_GENERATOR, False)
        self._tgClient.containers.run(f"{TAG}/influxdb:latest", name=INFLUXDB, detach=True, environment={
            "DOCKER_INFLUXDB_INIT_MODE": "setup",
            "DOCKER_INFLUXDB_INIT_USERNAME": INFLUX_DB_CONFIG["USERNAME"],
            "DOCKER_INFLUXDB_INIT_PASSWORD": INFLUX_DB_CONFIG["PASSWORD"],
            "DOCKER_INFLUXDB_INIT_ORG": INFLUX_DB_CONFIG["ORG"],
            "DOCKER_INFLUXDB_INIT_BUCKET": INFLUX_DB_CONFIG["BUCKET"],
            "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN": INFLUX_DB_CONFIG["TOKEN"]
        }, ports={"8086/tcp": 8086},
            volumes=[f"{TG_HOME_PATH}/tester/shared/data:/var/lib/influxdb2"])

        self._influxDBClient = InfluxDBClient(
            url="http://localhost:8086",
            token=INFLUX_DB_CONFIG["TOKEN"],
            org=INFLUX_DB_CONFIG["ORG"])

        NotificationSystem.subscribe(EMBEDDING_GRAPH_DEPLOYED, self)
        NotificationSystem.subscribe(EMBEDDING_GRAPH_DELETED, self)

    def setDesign(self, design: "list[TrafficDesign]") -> None:
        """
        Set the design of the traffic generator.

        Parameters:
            design (list[TrafficDesign]): The design of the traffic generator.
        """

        self._design = design

    def generateTraffic(self, sfcID: str) -> None:
        """
        Generate traffic.

        Parameters:
            sfcID (str): The ID of the SFC.
        """

        # Select a design randomly from the list of designs.
        random.seed()
        design: TrafficDesign = self._design[random.randint(
            0, len(self._design) - 1)]
        designStr: str = json.dumps(design)

        config: Config = getConfig()
        templateFile: str = f"{config['repoAbsolutePath']}/docker/files/k6/script.js.j2"
        outputFileContent: str = ""

        try:
            with open(templateFile, "r+", encoding="utf-8") as file:
                jTemplate: Template = Template(file.read())
                templatedFile = jTemplate.render(DESIGN=designStr)
                outputFileContent = templatedFile

        except FileNotFoundError:
            print(f"Template file {templateFile} not found.")
        except TemplateSyntaxError as e:
            print(
                f"Template syntax error in file {templateFile} on line {e.lineno}.")
            print(e)

        influxDBhost: str = self._tgClient.containers.get(
            INFLUXDB).attrs["NetworkSettings"]["IPAddress"]

        # pylint: disable=invalid-name
        name: str = f"{sfcID}-{K6}"
        self._tgClient.containers.run(f"{TAG}/k6:latest", cap_add="NET_ADMIN", name=name,
                                      detach=True, environment={
                                          "K6_OUT": f"xk6-influxdb=http://{influxDBhost}:8086",
                                          "K6_INFLUXDB_ORGANIZATION": INFLUX_DB_CONFIG["ORG"],
                                          "K6_INFLUXDB_BUCKET": INFLUX_DB_CONFIG["BUCKET"],
                                          "K6_INFLUXDB_INSECURE": "true",
                                          "K6_INFLUXDB_TOKEN": INFLUX_DB_CONFIG["TOKEN"]
                                      })
        self._tgClient.containers.get(name).exec_run(
            ["sh", "-c", f"echo '{outputFileContent}' > script.js"])
        self._tgClient.containers.get(name).exec_run(
            f"k6 run -e MY_HOSTNAME={getContainerIP(SFCC)} -e SFC_ID={sfcID} script.js", detach=True)

    def stopTrafficGeneration(self, sfcID: str) -> None:
        """
        Stop the traffic generation.

        Parameters:
            sfcID (str): The ID of the SFC.
        """

        self._tgClient.containers.get(f"{sfcID}-{K6}").remove(force=True)

    def getData(self, dataRange: str) -> "list[Any]":
        """
        Get the data from the traffic generator.

        Parameters:
            dataRange (str): Data range.

        Returns:
            "list[Any]": The data.
        """

        data: "list[Any]" = []
        tables = self._influxDBClient.query_api().query(
            f'from(bucket: "{INFLUX_DB_CONFIG["BUCKET"]}")'
            f' |> range(start: -{dataRange})'
            f' |> filter(fn: (r) => r["_measurement"] == "http_req_duration")'
            f' |> filter(fn: (r) => r["_field"] == "value")'
            f' |> filter(fn: (r) => r["expected_response"] == "true")'
            f' |> aggregateWindow(every: {dataRange}, fn: mean, createEmpty: false)'
            f' |> yield(name: "mean")')

        for table in tables:
            for record in table.records:
                data.append(record.values)

        return data

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == EMBEDDING_GRAPH_DEPLOYED:
            eg: EmbeddingGraph = args[0]
            self.generateTraffic(eg["sfcID"])
        elif topic == EMBEDDING_GRAPH_DELETED:
            eg: EmbeddingGraph = args[0]
            self.stopTrafficGeneration(eg["sfcID"])

    def __del__(self) -> None:
        """
        Destructor for the class.
        """

        self.end()

    def end(self) -> None:
        """
        End the traffic generator.
        """

        from_env().containers.get(TRAFFIC_GENERATOR).remove(force=True)
