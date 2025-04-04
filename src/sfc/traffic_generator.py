"""
Defines the TrafficGenerator class.
"""

import json
import random
from threading import Thread
from time import sleep
from typing import Any
from uuid import UUID, uuid4
from jinja2 import Template, TemplateSyntaxError
import pandas as pd
from shared.models.config import Config
from shared.models.embedding_graph import EmbeddingGraph
from shared.models.traffic_design import TrafficDesign
from shared.utils.config import getConfig
from influxdb_client import InfluxDBClient
from constants.container import DIND_IMAGE, TAG
from constants.notification import EMBEDDING_GRAPH_DELETED, EMBEDDING_GRAPH_DEPLOYED
from constants.topology import SFCC, TRAFFIC_GENERATOR
from mano.notification_system import NotificationSystem, Subscriber
from utils.container import connectToDind, getContainerIP, waitTillContainerReady
from docker import DockerClient, from_env, errors
from utils.tui import TUI
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)

TG_HOME_PATH: str = "/home/docker/files/influxdb"
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

    def __init__(self) -> None:
        """
        Constructor for the class.
        """

        self._design: "list[TrafficDesign]" = []
        self._tgClient: DockerClient = None
        self._influxDBClient: InfluxDBClient = None
        self._parentContainerStarted: bool = False
        self._sfcIdUUIDs: "dict[str, UUID]" = {}

        NotificationSystem.subscribe(EMBEDDING_GRAPH_DEPLOYED, self)
        NotificationSystem.subscribe(EMBEDDING_GRAPH_DELETED, self)

    def startParentContainer(self) -> None:
        """
        Start the parent container.
        """

        TUI.appendToLog("Starting traffic generator parent container.")
        config: Config = getConfig()
        dockerClient: DockerClient = from_env()

        try:
            dockerClient.containers.get(TRAFFIC_GENERATOR).remove(force=True)
        except errors.NotFound:
            pass

        try:
            dockerClient.containers.run(DIND_IMAGE, detach=True, name=TRAFFIC_GENERATOR, privileged=True, volumes=[
                f"{config['repoAbsolutePath']}/docker/files/influxdb:{TG_HOME_PATH}"
            ], command="dockerd", ports={"8086/tcp": 8086})
        except Exception as e:
            TUI.appendToLog(f"Error starting traffic generator parent container: {e}", True)

        TUI.appendToLog("Waiting for traffic generator parent container to start.")
        waitTillContainerReady(TRAFFIC_GENERATOR, False)
        TUI.appendToLog("Waiitng for traffic generator parent container to start is over.")

        self._tgClient = connectToDind(TRAFFIC_GENERATOR, False)
        try:
            self._tgClient.containers.run(f"{TAG}/influxdb:latest", name=INFLUXDB, detach=True, environment={
                "DOCKER_INFLUXDB_INIT_MODE": "setup",
                "DOCKER_INFLUXDB_INIT_USERNAME": INFLUX_DB_CONFIG["USERNAME"],
                "DOCKER_INFLUXDB_INIT_PASSWORD": INFLUX_DB_CONFIG["PASSWORD"],
                "DOCKER_INFLUXDB_INIT_ORG": INFLUX_DB_CONFIG["ORG"],
                "DOCKER_INFLUXDB_INIT_BUCKET": INFLUX_DB_CONFIG["BUCKET"],
                "DOCKER_INFLUXDB_INIT_ADMIN_TOKEN": INFLUX_DB_CONFIG["TOKEN"]
            }, ports={"8086/tcp": 8086},
                volumes=[f"{TG_HOME_PATH}/data:/var/lib/influxdb2"])
        except Exception as e:
            TUI.appendToLog(f"Error starting InfluxDB container: {e}", True)

        try:
            self._influxDBClient = InfluxDBClient(
                url="http://localhost:8086",
                token=INFLUX_DB_CONFIG["TOKEN"],
                org=INFLUX_DB_CONFIG["ORG"],
                timeout=int(config["general"]["requestTimeout"]) * 1000)
        except Exception as e:
            TUI.appendToLog(f"Error connecting to InfluxDB: {e}", True)

        TUI.appendToLog("Traffic generator parent container started.")
        self._parentContainerStarted = True

    def setDesign(self, design: "list[TrafficDesign]") -> None:
        """
        Set the design of the traffic generator.

        Parameters:
            design (list[TrafficDesign]): The design of the traffic generator.
        """

        self._design = design

    def _generateTraffic(self, sfcID: str) -> None:
        """
        Generate traffic.

        Parameters:
            sfcID (str): The ID of the SFC.
        """

        while not self._parentContainerStarted:
            TUI.appendToLog("Parent traffic generator container not started yet. Waiting for it to start.")
            sleep(1)

        TUI.appendToLog(f"Spinning up traffic generator for SFC {sfcID}:")
        # Select a design randomly from the list of designs.
        random.seed()
        index: int = random.randint(
            0, len(self._design) - 1)

        TUI.appendToLog(f"  Using traffic design {index}.")
        design: TrafficDesign = self._design[index]
        designStr: str = json.dumps(design)
        config: Config = getConfig()
        vus: int = config["k6"]["vus"]
        maxVus: int = config["k6"]["maxVus"]
        timeUnit: str = config["k6"]["timeUnit"]
        startRate: int = config["k6"]["startRate"]
        executor: str = config["k6"]["executor"]

        config: Config = getConfig()
        templateFile: str = f"{config['repoAbsolutePath']}/docker/files/k6/script.js.j2"
        outputFileContent: str = ""

        try:
            with open(templateFile, "r+", encoding="utf-8") as file:
                jTemplate: Template = Template(file.read())
                templatedFile = jTemplate.render(
                    DESIGN=designStr,
                    VUS=vus,
                    TIME_UNIT=timeUnit,
                    START_RATE=startRate,
                    EXECUTOR=executor,
                    MAX_VUS=maxVus,
                )
                outputFileContent = templatedFile

        except FileNotFoundError:
            TUI.appendToLog(f"  Template file {templateFile} not found.", True)
        except TemplateSyntaxError as e:
            TUI.appendToLog(
                f"  Template syntax error in file {templateFile} on line {e.lineno}.", True)
            TUI.appendToLog(f"  {e}", True)

        influxDBhost: str = self._tgClient.containers.get(
            INFLUXDB).attrs["NetworkSettings"]["IPAddress"]

        # pylint: disable=invalid-name
        uuid: UUID = uuid4()
        self._sfcIdUUIDs[sfcID] = uuid
        name: str = f"{sfcID}-{K6}-{uuid}"
        TUI.appendToLog(f"  Starting to generate traffic using k6 for SFC {sfcID}.")
        self._tgClient.containers.run(f"{TAG}/k6:latest", cap_add="NET_ADMIN", name=name,
                                    detach=True, environment={
                                        "K6_OUT": f"xk6-influxdb=http://{influxDBhost}:8086",
                                        "K6_INFLUXDB_ORGANIZATION": INFLUX_DB_CONFIG["ORG"],
                                        "K6_INFLUXDB_BUCKET": INFLUX_DB_CONFIG["BUCKET"],
                                        "K6_INFLUXDB_INSECURE": "true",
                                        "K6_INFLUXDB_TOKEN": INFLUX_DB_CONFIG["TOKEN"]
                                    })
        try:
            self._tgClient.containers.get(name).exec_run(
                ["sh", "-c", f"echo '{outputFileContent}' > script.js"])
        except Exception as e:
            # sometimes this exception is thrown: filedescriptor out of range in select().
            # seems like a Docker Py issue fixed by: https://github.com/docker/docker-py/pull/2865
            # since OpenRASE uses version 4.1.0, this doesn't include the fix.
            # Ignoring this issue as the script is created fine nevertheless.
            pass

        self._tgClient.containers.get(name).exec_run(
            f"k6 run -e MY_HOSTNAME={getContainerIP(SFCC)} -e SFC_ID={sfcID} script.js", detach=True)

        TUI.appendToLog(f"  Traffic generation started for SFC {sfcID}.")

    def _stopTrafficGeneration(self, sfcID: str) -> None:
        """
        Stop the traffic generation.

        Parameters:
            sfcID (str): The ID of the SFC.
        """

        try:
            self._tgClient.containers.get(f"{sfcID}-{K6}-{self._sfcIdUUIDs[sfcID]}").remove(force=True)
        except Exception as e:
            TUI.appendToLog(f"Error stopping traffic generator for SFC {sfcID}: {e}. Possibly because it has already been stopped.", True)

    def getData(self, dataRange: str) -> "pd.DataFrame":
        """
        Get the data from the traffic generator.

        Parameters:
            dataRange (str): Data range.

        Returns:
            pd.DataFrame: The data from the traffic generator.
        """

        HTTP_REQ_DURATION: str = "http_req_duration"

        return self._influxDBClient.query_api().query_data_frame(
            f'from(bucket: "{INFLUX_DB_CONFIG["BUCKET"]}")'
            f" |> range(start: -{dataRange})"
            f' |> filter(fn: (r) => r["_measurement"] == "{HTTP_REQ_DURATION}")'
            ' |> filter(fn: (r) => r["_field"] == "value")'
            ' |> filter(fn: (r) => r["expected_response"] == "true")'
            " |> map(fn: (r) => ({r with _time: uint(v: r._time)}))"
            ' |> drop(columns: ["_start", "_stop", "_measurement", "_field", "method", "name", "proto", "scenario", "status", "result", "table"])'
        )

    def receiveNotification(self, topic, *args: "list[Any]") -> None:
        if topic == EMBEDDING_GRAPH_DEPLOYED:
            egs: "list[EmbeddingGraph]" = args[0]
            for eg in egs:
                thread: Thread = Thread(target=self._generateTraffic, args=(eg["sfcID"],))
                thread.start()
        elif topic == EMBEDDING_GRAPH_DELETED:
            eg: EmbeddingGraph = args[0]
            self._stopTrafficGeneration(eg["sfcID"])

    def end(self) -> None:
        """
        End the traffic generator.
        """

        from_env().containers.get(TRAFFIC_GENERATOR).remove(force=True)

    def getDesign(self) -> "list[TrafficDesign]":
        """
        Get the design of the traffic generator.

        Returns:
            list[TrafficDesign]: The design of the traffic generator.
        """

        return self._design
