#!/usr/bin/env python3
"""Run the shared simcheck driver for this standalone specialization."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(directory.parent / "tools" / "simcheck.py"),
                "--case",
                directory.name,
                *sys.argv[1:],
            ]
        )
    )
