"""
Provides the encoding and decoding functions to encode/decode the SFC metadata in the HTTP headers.
"""

import json
import base64
from typing import Any, Union
from shared.models.embedding_graph import VNF


def sfcEncode(data: "Union[VNF, list[VNF]]") -> Any:
    """
    Encode the SFC metadata in the HTTP headers.

    Parameters:
        data (Union[VNF, list[VNF]]): SFC metadata to be encoded

    Returns:
        Any: Base64-encoded data
    """

    jsonData: str = json.dumps(data)
    base64Data: Any = base64.b64encode(jsonData.encode("ascii")).decode("ascii")

    return base64Data


def sfcDecode(data: Any) -> "Union[VNF, list[VNF]]":
    """
    Decode the SFC metadata in the HTTP headers.

    Parameters:
        data (Any): Base64-encoded data

    Returns:
        dict(Union[VNF, list[VNF]]): Decoded SFC metadata
    """

    jsonData: str = base64.b64decode(data.encode("ascii")).decode("ascii")
    dataDict: VNF = json.loads(jsonData)

    return dataDict
