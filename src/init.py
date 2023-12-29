"""
Initializes the emulator by building the Docker images of the VNFs, spinning up the registry
and configuring the settings.
"""

import json
import os
from typing import Any
from jinja2 import Template, TemplateSyntaxError
from docker import from_env, DockerClient, errors
from models.template_data import TemplateData
from shared.constants.sfc import SFC_REGISTRY
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.container import isContainerRunning

client: DockerClient = from_env()


def buildDockerImage(name: str) -> str:
    """
    Build a Docker image from a Dockerfile.

    Parameters:
        name (str): The name of the Dockerfile to build.
        This is the name of the directory in `docker/files/` that contains the Dockerfile.

    Returns:
        str: The name and tag of the Docker image that was built.
    """

    tag: str = f"{getRegistryContainerAddr()}/{name}:latest"

    client.images.build(
        path="./",
        dockerfile=f"docker/files/{name}/Dockerfile",
        tag=tag
    )

    return tag


def pushDockerImage(tag: str) -> None:
    """
    Push a Docker image to the registry.

    Parameters:
        tag (str): The tag of the Docker image to push.
    """

    client.images.push(tag)


def isRegistryContainerRunning() -> bool:
    """
    Check if the registry container is running.

    Returns:
        bool: True if the registry container is running, False otherwise.
    """

    return isContainerRunning(SFC_REGISTRY)


def startRegistryContainer() -> None:
    """
    Start the registry container.
    """

    client.containers.run(
        "registry:2",
        name=SFC_REGISTRY,
        ports={
            "5000/tcp": 5000
        },
        detach=True,
        environment={"REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY": "/data"},
        volumes={
            f"{getConfig()['repoAbsolutePath']}/docker/registry": {
                "bind": "/data",
                "mode": "rw"
            }}
    ).start()


def stopRegistryContainer() -> None:
    """
    Stop the registry container.
    """

    client.containers.get(SFC_REGISTRY).stop()
    client.containers.get(SFC_REGISTRY).remove()


def getRegistryContainerIP() -> str:
    """
    Get the IP address of the registry container.

    Returns:
        str: The IP address of the registry container.
    """

    return client.containers.get(SFC_REGISTRY).attrs["NetworkSettings"]["IPAddress"]


def getRegistryContainerAddr() -> str:
    """
    Get the tag of the registry container.

    Returns:
        str: The tag of the registry container.
    """

    return f"{getRegistryContainerIP()}:5000"


def addRegistryToInsecureRegistries() -> None:
    """
    Add the registry to the list of insecure registries
    in the Docker daemon.json file (/etc/docker/daemon.json).
    """

    addr: str = getRegistryContainerAddr()

    try:
        with open("/etc/docker/daemon.json", "r+", encoding="utf-8") as f:
            if not f.readable() or f.read().lstrip() == "":
                data: Any = {}
            else:
                f.seek(0)
                data: Any = json.load(f)

            if "insecure-registries" not in data:
                data["insecure-registries"] = [addr]
            elif addr not in data["insecure-registries"]:
                data["insecure-registries"].append(addr)

            f.seek(0)
            f.truncate()
            json.dump(data, f)
    except PermissionError:
        print("Permission denied. Please run this script as root or manually add\n" +
              f"{addr}\nto `insecure-registries` in /etc/docker/daemon.json.")


def generateTemplateData() -> TemplateData:
    """
    Generate the template data for the templates.
    This is a dictionary that contains the data te replace the placeholders in the templates.

    Returns:
        TemplateData: The template data.
    """

    config: Any = getConfig()

    return TemplateData(
        SFC_REGISTRY_TAG=getRegistryContainerAddr(),
        SFF_NETWORK1_IP=config["sff"]["network1"]["sffIP"],
        SFF_NETWORK2_IP=config["sff"]["network2"]["sffIP"],
        SFF_NETWORK1_NETWORK_IP=config["sff"]["network1"]["networkIP"],
        SFF_PORT=config["sff"]["port"]
    )


def generateConfigFilesFromTemplates() -> None:
    """
    Generate the config files from the templates.
    """

    config: Any = getConfig()
    templates: "list[str]" = config["templates"]

    for template in templates:
        mapper: "list[str]" = template.split(":")

        templateFile: str = f"{config['repoAbsolutePath']}/templates/{mapper[0]}"
        outputFile: str = mapper[1]

        templatedFile: str = ""
        try:
            with open(templateFile, "r+", encoding="utf-8") as file:
                jTemplate: Template = Template(file.read())
                templatedFile = jTemplate.render(generateTemplateData())

            with open(outputFile, "w", encoding="utf-8") as file:
                file.write(templatedFile)
        except FileNotFoundError:
            print(f"Template file {templateFile} not found.")
        except TemplateSyntaxError as e:
            print(
                f"Template syntax error in file {templateFile} on line {e.lineno}.")
            print(e)


def main() -> None:
    """
    The main function.
    """
    config: Config = getConfig()

    print("Initializing emulator...")

    print("Starting the registry container...")
    # Start registry container.
    if not isRegistryContainerRunning():
        startRegistryContainer()

    print("Adding registry to insecure registries...")
    # Add registry to insecure registries.
    addRegistryToInsecureRegistries()

    print("Generating config files from templates...")
    # Generate config files from templates.
    generateConfigFilesFromTemplates()

    print("Building and pushing Docker images...")
    # Build and push Docker images.
    for directory in os.listdir(f"{config['repoAbsolutePath']}/docker/files"):
        try:
            print("Building Docker image for " + directory)
            name: str = buildDockerImage(directory)
            print("Pushing Docker image " + name)
            pushDockerImage(name)
        except errors.DockerException as e:
            print("Docker image for " + directory +
                  " could not be built/pushed.")
            print(e)


main()
