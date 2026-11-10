"""Standalone Triton kernel for FPGA case 'quantize_transpose_128x32' (model 'gte-small').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in examples/BOSCAMEFPGA/gte-small/quantize_transpose_128x32
before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def quantize_transpose(
    X,
    Q,
    Scale,
    WIDTH: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    column = tl.program_id(0) * BC + tl.arange(0, BC)
    valid = (row[:, None] < WIDTH) & (column[None, :] < COLS)
    x = tl.load(X + row[:, None] * COLS + column[None, :], valid, other=0.0)
    x = tl.reshape(x, (BLOCK,))
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum == 0.0, 1.0, maximum / 127.0)
    value = tl.minimum(tl.maximum(tl.div_rn(x, scale), -127.0), 127.0)
    base = value.to(tl.int32)
    fraction = value - base.to(tl.float32)
    adjustment = tl.where(tl.abs(fraction) >= 0.5, tl.where(value >= 0, 1, -1), 0)
    integer = (base + adjustment).to(tl.int8)
    tl.store(Scale + tl.program_id(0), scale)
    tl.store(Q + tl.program_id(0) * WIDTH + row, integer, row < WIDTH)


KERNEL_CONFIG = {
    "family": "quantize_transpose",
    "kernel": "quantize_transpose",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Q", "rank": 2, "dtype": "i8"},
        {"name": "Scale", "rank": 1, "dtype": "f32"},
    ],
    "shape": [128, 32],
    "argument_shapes": [[128, 32], [32, 128], [32]],
    "constexprs": {"WIDTH": 128, "COLS": 32, "BLOCK": 128, "BC": 1},
    "grid": [32, 1, 1],
    "roles": [],
    "outputs": [1, 2],
    "checked_elements": 4128,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
