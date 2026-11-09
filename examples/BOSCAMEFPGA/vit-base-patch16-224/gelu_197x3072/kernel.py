"""
Standalone Triton kernel for FPGA case 'gelu_197x3072' (model
'vit-base-patch16-224').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/vit-base-patch16-224/gelu_197x3072 before linking launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def gelu(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """Exact Hugging Face GELU (the ViT config uses approximate="none")."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0).to(tl.float32)
    # tl.erf is lowered by triton-riscv to the target math implementation.
    y = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(Out + index, y, valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "gelu",
    "kernel": "gelu",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [197, 3072],
    "constexprs": {"COUNT": 605184, "BLOCK": 128},
    "grid": [4728, 1, 1],
    "roles": [],
    "math_accuracy": {
        "erf": {
            "domain": "all-f32",
            "special_values": "ieee",
            "max_abs_error": 2e-06,
            "max_rel_error": 0.0,
        },
    },
    "metadata": {"dtype": "f32", "model": "google/vit-base-patch16-224"},
}
