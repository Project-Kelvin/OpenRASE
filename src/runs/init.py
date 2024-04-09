"""
Initializes the emulator by building the Docker images of the VNFs, spinning up the registry
and configuring the settings.
"""

import json
import os
from typing import Any
from threading import Thread
from jinja2 import Template, TemplateSyntaxError
from shared.constants.sfc import SFC_REGISTRY
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.container import getRegistryContainerTag, isContainerRunning, doesContainerExist
from docker import from_env, DockerClient
from models.template_data import TemplateData

client: DockerClient = from_env()


def buildDockerImage(name: str) -> str:
    """
    Build a Docker image from a Dockerfile.

    Parameters:
        name (str): The name of the Dockerfile to build.
        This is the name of the directory in `docker/files/` that contains the Dockerfile.

    Returns:
        str: The name and tag of the Docker image that was built.
        An empty string if the image could not be built.
    """

    try:
        tag: str = f"{getRegistryContainerTag()}/{name}:latest"
        client.images.build(
            path=".",
            dockerfile=f"docker/files/{name}/Dockerfile",
            tag=tag
        )
    # pylint: disable=broad-except
    except Exception as e:
        print("Docker image for " + name +
              " could not be built.")
        print(e)

        return ""

    return tag


def pushDockerImage(tag: str) -> None:
    """
    Push a Docker image to the registry.

    Parameters:
        tag (str): The tag of the Docker image to push.
    """

    try:
        client.images.push(tag)
    except ConnectionError as e:
        print("Docker image " + tag + " could not be pushed.")
        print(e)


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

    if doesContainerExist(SFC_REGISTRY):
        client.containers.get(SFC_REGISTRY).remove()

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
            }},
        auto_remove=True
    ).start()


def stopRegistryContainer() -> None:
    """
    Stop the registry container.
    """

    client.containers.get(SFC_REGISTRY).stop()
    client.containers.get(SFC_REGISTRY).remove()


def addRegistryToInsecureRegistries() -> None:
    """
    Add the registry to the list of insecure registries
    in the Docker daemon.json file (/etc/docker/daemon.json).
    """

    addr: str = getRegistryContainerTag()

    try:
        with open("/etc/docker/daemon.json", "w+", encoding="utf-8") as f:
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
    This is a dictionary that contains the data to replace the placeholders in the templates with.

    Returns:
        TemplateData: The template data.
    """

    config: Any = getConfig()

    return TemplateData(
        SFC_REGISTRY_TAG=getRegistryContainerTag(),
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

        templateFile: str = os.path.join(config['repoAbsolutePath'], "templates", mapper[0])
        outputFile: str = os.path.join(config['repoAbsolutePath'], mapper[1])
        templatedFile: str = ""
        try:
            with open(templateFile, "r+", encoding="utf-8") as file:
                jTemplate: Template = Template(file.read())
                templatedFile = jTemplate.render(generateTemplateData())

            with open(outputFile, "w+", encoding="utf-8") as file:
                file.write(templatedFile)
        except FileNotFoundError:
            print(f"Template file {templateFile} not found.")
        except TemplateSyntaxError as e:
            print(
                f"Template syntax error in file {templateFile} on line {e.lineno}.")
            print(e)


def symLinkConfig() -> None:
    """
    Create a symlink to the config file in every app directory.
    """

    config: Config = getConfig()

    try:
        for directory in os.listdir(f"{config['repoAbsolutePath']}/apps/"):
            os.symlink(
                f"{config['repoAbsolutePath']}/config.yaml",
                f"{config['repoAbsolutePath']}/apps/{directory}/config.yaml"
            )
    except FileExistsError:
        pass

def createArtifactsDirectory() -> None:
    """
    Create the artifacts directory.
    """

    config: Config = getConfig()

    try:
        os.mkdir(f"{config['repoAbsolutePath']}/artifacts")
    except FileExistsError:
        pass


def run() -> None:
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

    print("Creating symlinks to config file...")
    # Create symlink to config file.
    symLinkConfig()

    print("Building and pushing Docker images...")
    # Build and push Docker images.

    threads: "list[Thread]" = []
    for directory in os.listdir(f"{config['repoAbsolutePath']}/docker/files"):
        def buildAndPush(directory: str):
            print("Building Docker image for " + directory)
            name: str = buildDockerImage(directory)
            if name != "":
                print("Pushing Docker image " + name)
                pushDockerImage(name)

        thread: Thread = Thread(target=buildAndPush, args=(directory,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("Creating artifacts directory...")
    # Create artifacts directory.
    createArtifactsDirectory()
