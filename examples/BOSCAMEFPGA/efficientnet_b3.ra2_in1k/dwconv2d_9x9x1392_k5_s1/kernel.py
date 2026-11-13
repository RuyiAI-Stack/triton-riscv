"""
Standalone Triton kernel for FPGA case 'dwconv2d_9x9x1392_k5_s1' (model
'efficientnet_b3.ra2_in1k').

Inputs: pointer arguments and KERNEL_CONFIG constexprs (BLOCK, COUNT, and tensor
shapes) for AOT compile.

Outputs: an @triton.jit entry plus KERNEL_CONFIG metadata consumed by contract
export and buddy-opt.

When used: Make / export.py in
examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/dwconv2d_9x9x1392_k5_s1 before
linking launch.c.
"""

KERNEL_CONFIG = {
    "family": "dwconv2d_f32",
    "implementation": "im2col_w8a8_depthwise",
    "kernel": "dwconv2d_im2col_w8a8",
    "arguments": [
        {"name": "X", "rank": 4, "dtype": "f32"},
        {"name": "Weight", "rank": 4, "dtype": "f32"},
        {"name": "Bias", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 4, "dtype": "f32"},
    ],
    "shape": [1, 1392, 9, 9],
    "constexprs": {
        "H": 9,
        "W": 9,
        "C": 1392,
        "KH": 5,
        "KW": 5,
        "SH": 1,
        "SW": 1,
        "PH": 2,
        "PW": 2,
        "OH": 9,
        "OW": 9,
    },
    "grid": [1, 1, 1],
    "roles": [],
    "metadata": {"dtype": "efficientnet-b3", "model": "timm/efficientnet_b3.ra2_in1k"},
    "kernel_module": "quantized_conv_kernels",
}
