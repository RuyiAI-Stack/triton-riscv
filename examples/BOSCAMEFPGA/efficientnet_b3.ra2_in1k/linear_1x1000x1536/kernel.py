"""
Standalone Triton kernel for FPGA case 'linear_1x1000x1536' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/linear_1x1000x1536 before linking
launch.c.
"""

KERNEL_CONFIG = {
    "family": "linear_f32",
    "kernel": "linear_w8a8",
    "arguments": [
        {"name": "A", "rank": 2, "dtype": "f32"},
        {"name": "W", "rank": 2, "dtype": "f32"},
        {"name": "Bias", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [1, 1000, 1536],
    "constexprs": {"M": 1, "N": 1000, "K": 1536, "BM": 1, "BN": 16, "BK": 64},
    "grid": [1, 63, 1],
    "roles": [],
    "metadata": {"dtype": "efficientnet-b3", "model": "timm/efficientnet_b3.ra2_in1k"},
    "implementation": "w8a8_gemm",
    "kernel_module": "quantized_linear_kernels",
}
