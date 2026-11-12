"""
Standalone Triton kernel for FPGA case 'gap_9x9x1392' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/gap_9x9x1392 before linking
launch.c.
"""

import triton
import triton.language as tl


@triton.jit
def gap_scalar_reference(
    X, Out, SPATIAL: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr
):
    """Global average pool. One program reduces one channel along unit-stride pixels."""
    tl.static_assert(C > 0)
    ch = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for block in range(tl.cdiv(SPATIAL, BLOCK)):
        pixel = block * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(X + ch * SPATIAL + pixel, pixel < SPATIAL, other=0.0)
    tl.store(Out + ch, tl.div_rn(tl.sum(acc), SPATIAL * 1.0))


# Fixed specialization; cases.py reads this literal without importing Triton.
KERNEL_CONFIG = {
    "family": "gap",
    "kernel": "gap_phase_ordered",
    "implementation": "hybrid_ordered_c",
    "implementation_source": "diagnostics/model_integration/gap_torch_order_candidate.c",
    "implementation_symbol_prefix": "efficientnet_phase_",
    "kernel_module": "gap_phase_kernels",
    "reference_kernel": "gap_scalar_reference",
    "arguments": [
        {"name": "X", "rank": 4, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [1, 1392, 9, 9],
    "constexprs": {"SPATIAL": 81, "C": 1392, "BLOCK": 128},
    "grid": [1392, 1, 1],
    "roles": [],
    "metadata": {"dtype": "efficientnet-b3", "model": "timm/efficientnet_b3.ra2_in1k"},
}
