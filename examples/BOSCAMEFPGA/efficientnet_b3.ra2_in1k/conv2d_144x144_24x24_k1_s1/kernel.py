"""
Standalone Triton kernel for FPGA case 'conv2d_144x144_24x24_k1_s1' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/conv2d_144x144_24x24_k1_s1 before
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
    "shape": [1, 24, 144, 144],
    "constexprs": {
        "H": 144,
        "W": 144,
        "C": 24,
        "OC": 24,
        "KH": 1,
        "KW": 1,
        "SH": 1,
        "SW": 1,
        "PH": 0,
        "PW": 0,
        "OH": 144,
        "OW": 144,
    },
    "grid": [1, 1, 1],
    "roles": [],
    "metadata": {"dtype": "efficientnet-b3", "model": "timm/efficientnet_b3.ra2_in1k"},
    "kernel_module": "quantized_conv_kernels",
}
