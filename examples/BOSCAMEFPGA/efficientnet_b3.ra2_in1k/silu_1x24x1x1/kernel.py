"""
Standalone Triton kernel for FPGA case 'silu_1x24x1x1' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/silu_1x24x1x1 before linking
launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def silu_scalar_reference(X, Out, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    """SiLU as x / (1 + exp(-x)), with an authorized exp approximation."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < COUNT
    x = tl.load(X + index, valid, other=0.0)
    tl.store(Out + index, tl.div_rn(x, 1.0 + tl.exp(-x)), valid)


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "silu",
    "implementation": "hybrid_rvv_c",
    "implementation_source": "diagnostics/model_integration/silu_rvv_direct_output_candidate.c",
    "implementation_symbol_prefix": "efficientnet_rvv_",
    "kernel_module": "silu_rvv_kernels",
    "reference_kernel": "silu_scalar_reference",
    "kernel": "silu_rvv",
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
    "shape": [1, 24, 1, 1],
    "constexprs": {
        "COUNT": 24,
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
