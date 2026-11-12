#!/usr/bin/env python3
"""Generate typed ABI bridges without importing Triton or compiling a kernel."""

import argparse
from pathlib import Path
from export import ROOT, inventory, adapter_source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=ROOT / "build/kernels")
args = parser.parse_args()
for case in inventory():
    directory = args.output / case["name"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter.c").write_text(adapter_source(case))
