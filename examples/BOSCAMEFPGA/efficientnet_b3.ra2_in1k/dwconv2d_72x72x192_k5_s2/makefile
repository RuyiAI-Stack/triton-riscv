#
# Per-case Make rules for FPGA operator 'dwconv2d_72x72x192_k5_s2' (model
# 'efficientnet_b3.ra2_in1k').
#
# Inputs: this directory's kernel.py / launch.c plus parent common.mk compiler
# stages.
#
# Outputs: AOT artifacts and the board image for this shape only, not sibling
# cases.
#
# When used: `make` inside the case directory or via the model top-level
# makefile.
#
CONV_LLVM_SCALAR_O2 := 1
CONV_PATCH_DDR_PHASES := 1
include ../common.mk
