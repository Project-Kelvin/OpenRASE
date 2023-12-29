"""
Defines the models related to the SFF app.
"""

from typing import TypedDict

class AddHostRequestBody(TypedDict):
    """
    Defines the `AddHostRequestBody` dictionary type.
    """

    hostIP: str
