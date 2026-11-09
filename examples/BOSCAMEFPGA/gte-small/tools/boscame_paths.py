"""Path helpers for BOSCAME FPGA model 'gte-small'.

Inputs: this file location; optional env overrides for buddy-opt and repo roots.

Outputs: resolved paths to triton-riscv, buddy-mlir, and build directories.

When used: export.py, simcheck, and report scripts so they do not hard-code
checkouts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from paths import (  # noqa: E402, F401
    buddy_tools,
    llvm_lib_dir,
    source_identity,
)
