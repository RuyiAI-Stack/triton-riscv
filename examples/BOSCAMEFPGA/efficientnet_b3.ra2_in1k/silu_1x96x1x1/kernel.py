"""Standalone Triton source for silu_1x96x1x1."""

import triton
import triton.language as tl

@triton.jit
def silu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """SiLU as x / (1 + exp(-x)), with an authorized exp approximation."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    tl.store(Out + index, tl.div_rn(x, 1.0 + tl.exp(-x)), valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "silu",
    "kernel": "silu",
    "arguments": [
        {
            "name": "X",
            "rank": 4,
            "dtype": "f32",
        },
        {
            "name": "Out",
            "rank": 4,
            "dtype": "f32",
        },
    ],
    "shape": [1, 96, 1, 1],
    "constexprs": {
        "COUNT": 96,
        "BLOCK": 128,
    },
    "grid": [1, 1, 1],
    "roles": [],
    "math_accuracy": {
        "exp": {
            "domain": "all-f32",
            "special_values": "ieee",
            "max_abs_error": 1.1210387714598537e-44,
            "max_rel_error": 2e-06,
        },
    },
    "metadata": {
        "dtype": "efficientnet-b3",
        "model": "timm/efficientnet_b3.ra2_in1k",
    },
}
