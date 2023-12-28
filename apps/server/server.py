"""
A simple web server based on Flask that returns "Hello World!"
or the first argument passed to the script.
"""

import sys
from flask import Flask

from shared.models.config import Config
from shared.utils.config import getConfig

app: Flask = Flask(__name__)


@app.route("/tx")
def default() -> str:
    """ Default endpoint that returns "Hello World!" or the first argument passed to the script."""

    if len(sys.argv) > 1 and sys.argv[1] != "" and sys.argv[1] is not None:
        return sys.argv[1]
    else:
        return "Hello World!\n"


config: Config = getConfig()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config["server"]["port"])
