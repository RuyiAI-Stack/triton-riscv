#!/usr/bin/env python3
"""Test bridge: external TTIR frontend followed by the independent Buddy tool."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    args = sys.argv[1:]
    buddy = os.environ.get("BUDDY_OPT") or shutil.which("buddy-opt")
    if not buddy:
        raise SystemExit("set BUDDY_OPT to the independent Buddy compiler")
    frontend = [
        arg for arg in args if arg.startswith("--triton-to-linalg-experimental")
    ]
    if not frontend:
        return subprocess.run([buddy, *args]).returncode
    if len(frontend) != 1:
        raise SystemExit("expected one external frontend pass")
    source = None
    output = None
    backend = []
    common = []
    cursor = 0
    while cursor < len(args):
        arg = args[cursor]
        if arg in frontend:
            pass
        elif arg == "-o":
            cursor += 1
            output = args[cursor]
        elif arg == "-" or not arg.startswith("-"):
            if source is not None:
                raise SystemExit("expected one input file")
            source = arg
        else:
            backend.append(arg)
            if arg.startswith("--mlir-"):
                common.append(arg)
        cursor += 1
    tool = os.environ.get("TRITON_SHARED_OPT_PATH")
    if not tool:
        raise SystemExit(
            "integration tests require explicit TRITON_SHARED_OPT_PATH"
        )
    with tempfile.TemporaryDirectory(
        prefix="contract-integration-"
    ) as temporary:
        boundary = Path(temporary) / "contract.mlir"
        command = [tool, source or "-", *frontend, *common, "-o", str(boundary)]
        if "--split-input-file" in backend:
            command.append("--split-input-file")
        result = subprocess.run(command)
        if result.returncode:
            return result.returncode
        command = [buddy, str(boundary), *backend]
        if output:
            command.extend(["-o", output])
        return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())
