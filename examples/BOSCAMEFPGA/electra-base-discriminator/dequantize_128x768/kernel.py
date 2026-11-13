"""Standalone Triton kernel for FPGA case 'dequantize_128x768' (model 'electra-base-discriminator').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/electra-base-discriminator/dequantize_128x768
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def dequantize(
    X,
    Row,
    Column,
    Out,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    FACTOR: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < COLS
    index = row * COLS + column
    value = tl.load(X + index, valid, other=0).to(tl.float32)
    rs = tl.load(Row + row)
    cs = tl.load(Column + column, valid, other=1.0)
    tl.store(Out + index, ((value * rs) * cs) * FACTOR, valid)


KERNEL_CONFIG = {
    "family": "dequantization",
    "kernel": "dequantize",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "i32"},
        {"name": "Row", "rank": 1, "dtype": "f32"},
        {"name": "Column", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [128, 768],
    "constexprs": {"ROWS": 128, "COLS": 768, "BLOCK": 1024, "FACTOR": 1.0},
    "grid": [128, 1, 1],
    "roles": [],
    "metadata": {
        "dtype": "i32 -> f32",
        "target": "scalar",
        "model": "google/electra-base-discriminator",
    },
    "output_factor": 1.0,
}
