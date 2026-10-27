"""Standalone Triton source for softmax_12x197x197."""

import triton
import triton.language as tl


@triton.jit
def softmax(
    X, Out, ROWS: tl.constexpr, LENGTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    valid = column < LENGTH
    x = tl.load(X + row * LENGTH + column, valid, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    total = tl.sum(exponential, axis=0)
    tl.store(Out + row * LENGTH + column, exponential / total, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "softmax",
    "kernel": "softmax",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "shape": [12, 197, 197],
    "constexprs": {"ROWS": 2364, "LENGTH": 197, "BLOCK": 256},
    "grid": [2364, 1, 1],
    "roles": [],
    "math_accuracy": {
        "exp": {
            "domain": "all-f32",
            "special_values": "ieee",
            "max_abs_error": 1.1210387714598537e-44,
            "max_rel_error": 2e-06,
        },
    },
    "metadata": {"dtype": "f32", "model": "google/vit-base-patch16-224"},
}
