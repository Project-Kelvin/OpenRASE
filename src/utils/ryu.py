"""
Defines the util functions associated with the Ryu controller.
"""

from constants.ryu import RYU_REST_URL

def getRyuRestUrl(switchID: str) -> str:
    """
    Get the URL of the Ryu REST API.

    Parameters:
    switchID (str): The ID of the switch.

    Returns:
    str: The URL of the Ryu REST API.
    """


    return f"{RYU_REST_URL}/{switchID}"
