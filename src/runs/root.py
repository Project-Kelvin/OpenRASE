"""
Defines the script for running the root service for the HiGENESIS algorithm.
"""

import os
import subprocess
import sys

from shared.utils.config import getConfig

def run() -> None:
    """
    Starts the root service for the HiGENESIS algorithm.
    """

    rootDir: str = getConfig()["repoAbsolutePath"]
    script: str = os.path.join(rootDir, "src", "algorithms", "hybrid", "utils", "root", "root_service.py")
    cmd: list[str] = [sys.executable, script]
    subprocess.run(
        cmd,
        check=True,
    )
