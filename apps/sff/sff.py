"""
A Service Function Forwarder (SFF) that receives traffic to the host it runs on
and forwards it to the VNF, and transmits the traffic from the VNF to the next
SFF/VNF in the Service Function Chain.
"""

from wsgiref.headers import Headers

from shared.constants.forwarding_graph import TERMINAL
from shared.models.config import Config
from shared.models.forwarding_graph import VNF
from shared.constants.sfc import SFC_HEADER, SFC_TRAVERSED_HEADER
from shared.utils.encoder_decoder import sfcDecode, sfcEncode
from shared.utils.config import getConfig
from shared.utils.ip import checkIPBelongsToNetwork
from flask import Flask, Response, request, Request
import requests


app: Flask = Flask(__name__)
config: Config = getConfig()

# In-memory list of host IP addresses.
hosts: "list[str]" = []


def extractAndValidateSFCHeader(requestObj: Request) -> VNF:
    """
    Extract and validate the SFC header from the request.

    Parameters:
        request (Request): The request received by the SFF.

    Returns:
        dict: The SFC metadata.

    Raises:
        Exception: If the SFC header is not found in the request.
        Exception: If the request is sent to the wrong host.
    """

    sfcBase64: str = request.headers.get(SFC_HEADER)

    if len(hosts) == 0:
        raise RuntimeError("This SFF has no hosts assigned to it.\n" +
                                        "Please add the host IP address using the `/add-host` endpoint.\n")

    if sfcBase64 == "" or sfcBase64 is None:
        raise ValueError(f"The SFF running on\n {', '.join(hosts)}\n" +
                                        " could not find the SFC header " +
                                        f"attribute request from:\n{requestObj.host_url}.\n")
    sfc: VNF = sfcDecode(sfcBase64)

    if sfc["host"]["ip"] not in hosts:
        raise RuntimeError("This request arrived at the wrong host.\n" +
                                        "This host has the following IP addresses:\n" +
                                        f"{', '.join(hosts)}.\n" +
                                        f"However, this request was sent to {sfc['host']}.\n")

    return sfc


@app.route("/rx", strict_slashes=False)
def rx() -> Response:
    """
    The endpoint that receives the traffic from the previous SFF and forwards it to the next VNF.
    """

    try:
        sfc: VNF = extractAndValidateSFCHeader(request)

        if "isTraversed" in sfc and sfc["isTraversed"] is True:
            return f"VNF {sfc['vnf']['id']} has already processed this request.", 400

        vnfIP: str = sfc["vnf"]["ip"]

        response: Response = requests.request(
            method=request.method,
            url=request.url.replace(f'{request.host_url}rx', f"http://{vnfIP}"),
            headers=request.headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=5
        )

        return Response(response, status=response.status_code)

    # pylint: disable=broad-except
    except Exception as e:
        return str(e), 400


@app.route("/tx", strict_slashes=False)
def tx() -> Response:
    """
    The endpoint that receives the traffic from the previous VNF
    and forwards it to the next SFF/VNF.
    """

    try:
        sfc: VNF = extractAndValidateSFCHeader(request)

        # Gets the `SFC-Traversed` header from the request.
        sfcTraversed: "list[VNF]" = request.headers.get(SFC_TRAVERSED_HEADER)
        if sfcTraversed != "" and sfcTraversed is not None:
            sfcTraversed = sfcDecode(sfcTraversed)
        else:
            sfcTraversed = []

        # Sets the `isTraversed` attribute of the current VNF to True.
        sfc["isTraversed"] = True
        sfcUpdated: VNF = sfc.copy()
        del sfcUpdated["next"]
        sfcTraversed.append(sfcUpdated)

        # Handles branching of SFC.
        if isinstance(sfc["next"], list):
            network1IP: str = getConfig()["sff"]["network1"]["networkIP"]

            if checkIPBelongsToNetwork(request.remote_addr, network1IP):
                sfc = sfc["next"][0]
            else:
                sfc = sfc["next"][1]
        else:
            sfc = sfc["next"]

        # Makes forwarding decision.
        nextDest: str = ""
        if sfcUpdated["host"]["id"] == sfc["host"]["id"]:
            nextDest = f"http://{sfc['vnf']}"
        elif sfc["next"] == TERMINAL:
            nextDest = f'http://{sfc["host"]["ip"]}'
        else:
            nextDest = f'http://{sfc["host"]["ip"]}/rx'

        # Updates header.
        sfc["isTraversed"] = False
        headers: Headers = {}
        for key, value in request.headers.items():
            headers[key] = value

        sfcBase64: str = sfcEncode(sfc)
        sfcTraversed: str = sfcEncode(sfcTraversed)
        headers[SFC_HEADER] = sfcBase64
        headers[SFC_TRAVERSED_HEADER] = sfcTraversed

        response = requests.request(
            method=request.method,
            url=request.url.replace(f'{request.host_url}tx', nextDest),
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=config["general"]["requestTimeout"]
        )

        return Response(response, status=response.status_code)

    # pylint: disable=broad-except
    except Exception as e:
        return str(e), 400


@app.route("/add-host", methods=["POST"], strict_slashes=False)
def addHost() -> Response:
    """
    The endpoint that adds the IP address assigned to the host the SFF is running on
    to an in-memory list.
    """

    ipAddress: str = request.json["hostIP"]

    hosts.append(ipAddress)

    return Response(status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["sff"]["port"])
