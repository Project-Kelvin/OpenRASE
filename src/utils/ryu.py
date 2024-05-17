"""
Defines the util functions associated with the Ryu controller.
"""

from constants.ryu import RYU_REST_URL

def getRyuRestUrl() -> str:
    """
    Get the URL of the Ryu REST API.

    Returns:
        str: The URL of the Ryu REST API.
    """


    return RYU_REST_URL
