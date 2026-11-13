#!/usr/bin/env python3
"""Compile the current convolution pipeline into an isolated final artifact set."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from export import inventory  # noqa: E402


def main():
    output = ROOT / "build/w8a8-opt/precleared-integrated"

    def one(case):
        directory = output / case["name"]
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "make.log").open("w") as log:
            process = subprocess.run(
                [
                    "make",
                    "-s",
                    "-C",
                    case["directory"],
                    "bin",
                    "KERNELS=" + str(output),
                ],
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            raise RuntimeError(
                case["name"] + "\n" + (directory / "make.log").read_text()[-2500:]
            )
        print("PASS compiled", case["name"], flush=True)

    cases = [
        case for case in inventory() if case["family"] in ("conv2d_f32", "dwconv2d_f32")
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, cases))


if __name__ == "__main__":
    main()
