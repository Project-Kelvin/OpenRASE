"""
Contains the model of the Jinja2 template data dictionary.
"""

from typing import TypedDict


class TemplateData(TypedDict):
    """
    Defines the `TemplateData` dictionary type.
    """
    # pylint: disable=invalid-name
    SFC_REGISTRY_IP: str
    SFF_NETWORK1_IP: str
    SFF_NETWORK2_IP: str
    SFF_PORT: str
    SFF_NETWORK1_NETWORK_IP: str
