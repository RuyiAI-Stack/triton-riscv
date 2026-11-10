"""
Standalone Triton kernel for FPGA case 'conv2d_288x288_3x40_k3_s2' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/conv2d_288x288_3x40_k3_s2 before
linking launch.c.
"""

KERNEL_CONFIG = {
    "family": "conv2d_f32",
    "implementation": "im2col_w8a8",
    "kernel": "conv2d_im2col_w8a8",
    "arguments": [
        {"name": "X", "rank": 4, "dtype": "f32"},
        {"name": "Weight", "rank": 4, "dtype": "f32"},
        {"name": "Bias", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 4, "dtype": "f32"},
    ],
    "shape": [1, 3, 288, 288],
    "constexprs": {
        "H": 288,
        "W": 288,
        "C": 3,
        "OC": 40,
        "KH": 3,
        "KW": 3,
        "SH": 2,
        "SW": 2,
        "PH": 1,
        "PW": 1,
        "OH": 144,
        "OW": 144,
    },
    "grid": [1, 1, 1],
    "roles": [],
    "metadata": {"dtype": "efficientnet-b3", "model": "timm/efficientnet_b3.ra2_in1k"},
    "kernel_module": "quantized_conv_kernels",
}
