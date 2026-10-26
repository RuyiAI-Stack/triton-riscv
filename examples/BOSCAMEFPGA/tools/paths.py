"""Explicit external Buddy paths for optional BOSCAME backend workflows."""

import os
import shutil
from pathlib import Path

FRONTEND_ROOT = Path(__file__).resolve().parents[3]


def buddy_root():
    value = os.environ.get("BUDDY_ROOT")
    if not value:
        raise RuntimeError("set BUDDY_ROOT to the external Buddy checkout")
    root = Path(value).resolve()
    if not (root / "runtime/boscame").is_dir():
        raise RuntimeError("BUDDY_ROOT does not contain runtime/boscame")
    return root


def buddy_tools():
    return buddy_root() / "examples/FPGA-BOSCAME/tools"


def source_identity(path):
    """Name evidence sources without embedding checkout-specific paths."""
    path = Path(path).resolve()
    if path.is_relative_to(FRONTEND_ROOT):
        return "triton-riscv/" + path.relative_to(FRONTEND_ROOT).as_posix()
    return "buddy-mlir/" + path.relative_to(buddy_root()).as_posix()


def llvm_lib_dir():
    if value := os.environ.get("LLVM_LIB_DIR"):
        return Path(value)
    if value := os.environ.get("LLVM_BIN"):
        return Path(value).resolve().parent / "lib"
    if tool := shutil.which("mlir-runner"):
        return Path(tool).resolve().parent.parent / "lib"
    return None
