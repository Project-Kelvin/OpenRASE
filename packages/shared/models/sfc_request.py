"""
Defines the models associated with SFC Requests.
"""

from typing import TypedDict


class SFCRequest(TypedDict):
    """
    Defines the `SFCRequest` dictionary type.
    """

    sfcrID: str
    latency: int
    vnfs: "list[str]"
    strictOrder: "list[str]"
