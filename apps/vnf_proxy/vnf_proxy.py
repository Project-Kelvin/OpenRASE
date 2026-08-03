"""
Proxies all request to the VNF from the SFF back to the `/tx` endpoint of the SFF.
This is useful in passive VNFs that just listens to the traffic on a port.
"""

from flask import Flask, Response, request
import requests
from shared.models.config import Config

from shared.utils.config import getConfig

app: Flask = Flask(__name__)
config: Config = getConfig()


@app.route("/")
def default():
    """ Default endpoint that proxies requests to the SFF. """

    sffIP: str = getConfig()["sff"]["network1"]["sffIP"]
    sffPort: str = getConfig()["sff"]["port"]

    response = requests.request(
        method=request.method,
        url=request.url.replace(request.host_url, f'http://{sffIP}:{sffPort}/tx'),
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        timeout=config["general"]["requestTimeout"],
        headers=request.headers
    )

    return Response(response, status=response.status_code)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["vnfProxy"]["port"])
