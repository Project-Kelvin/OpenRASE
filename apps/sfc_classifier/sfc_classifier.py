"""
A Service Function Chain (SFC) Classifier that classifies the incoming traffic into different SFCs
using the source IP address of the traffic.

Following the classification, the SFC Classifier adds the SFC metadata to the HTTP headers and forwards
the traffic to the next SFF/VNF in the SFC.
"""

from typing import Any, TypedDict
from wsgiref.headers import Headers
from flask import Flask, Response, request
import requests
from shared.constants.sfc import SFC_HEADER, SFC_ID
from shared.models.forwarding_graph import VNF, ForwardingGraph, ForwardingGraphs
from shared.models.config import Config
from shared.utils.config import getConfig
from shared.utils.encoder_decoder import sfcEncode

app: Flask = Flask(__name__)
config: Config = getConfig()

forwardingGraphs: "TypedDict[str, ForwardingGraphs]" = {}
fgLock: bool = False


def addForwardingGraphToMemory(sfcID: str, forwardingGraph: ForwardingGraph):
    """
    Add the VNF Forwarding Graph to the in-memory list of VNF Forwarding Graphs.
    """

    # pylint: disable=global-statement
    global fgLock

    while fgLock is True:
        pass

    fgLock = True
    forwardingGraphs[sfcID] = forwardingGraph
    fgLock = False


@app.route("/add-fg", methods=['POST'], strict_slashes=False)
def addFG():
    """
    The endpoint that receives the Forwarding Graph as a JSON object and
    adds it to the in-memory list of Forwarding Graphs.
    """

    fg: ForwardingGraph = request.get_json()
    fg["isTraversed"] = False

    addForwardingGraphToMemory(fg["sfcID"], fg)

    return "The Forwarding Graph has been successfully added.\n", 201


@app.route("/")
def default():
    """
    Default endpoint that classifies SFC requests adds the SFC metadata to the request headers.
    """

    try:
        if SFC_ID not in request.headers:
            return "The SFC-ID Header is missing in the request.\n", 400

        sfcID: str = request.headers[SFC_ID]

        if sfcID not in forwardingGraphs:
            return (
                "The SFC-ID is not registered with the SFC Classifier.\n"
                "Use the `add-fg` endpoint to register it."
            ), 400

        sfc: VNF = forwardingGraphs[sfcID]["vnfs"]

        sfcBase64: Any = sfcEncode(sfc)

        headers: Headers = {}
        for key, value in request.headers.items():
            headers[key] = value

        headers[SFC_HEADER] = sfcBase64

        response = requests.request(
            method=request.method,
            url=request.url.replace(
                request.host_url, f'http://{sfc["host"]["ip"]}/rx'),
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=config["general"]["requestTimeout"],
            headers=headers
        )

        return Response(response, status=response.status_code)

    # pylint: disable=broad-except
    except Exception as exception:
        return str(exception), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["sfcClassifier"]["port"])
