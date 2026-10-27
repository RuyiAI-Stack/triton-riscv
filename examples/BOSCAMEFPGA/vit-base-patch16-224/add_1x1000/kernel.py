"""Standalone Triton source for add_1x1000."""

import triton
import triton.language as tl


@triton.jit
def add(X, Y, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    tl.store(
        Out + index,
        tl.load(X + index, valid, other=0.0)
        + tl.load(Y + index, valid, other=0.0),
        valid,
    )


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "add",
    "kernel": "add",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Y", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [1, 1000],
    "constexprs": {"COUNT": 1000, "BLOCK": 128},
    "grid": [8, 1, 1],
    "roles": [],
    "metadata": {"dtype": "f32", "model": "google/vit-base-patch16-224"},
}
