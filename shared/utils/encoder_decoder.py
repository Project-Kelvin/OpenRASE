"""
Provides the encoding and decoding functions to encode/decode the SFC metadata in the HTTP headers.
"""

import json
import base64
from typing import Any
from shared.models.forwarding_graph import VNF


def sfcEncode(data: VNF) -> Any:
    """
    Encode the SFC metadata in the HTTP headers.

    Parameters:
        data (dict): SFC metadata to be encoded

    Returns:
        Any: Base64-encoded data
    """

    jsonData: str = json.dumps(data)
    base64Data: Any = base64.b64encode(jsonData.encode("ascii")).decode("ascii")

    return base64Data


def sfcDecode(data: Any) -> VNF:
    """
    Decode the SFC metadata in the HTTP headers.

    Parameters:
        data (Any): Base64-encoded data

    Returns:
        dict: Python Dictionary containing the decoded SFC metadata
    """

    jsonData: str = base64.b64decode(data.encode("ascii")).decode("ascii")
    dataDict: VNF = json.loads(jsonData)

    return dataDict
