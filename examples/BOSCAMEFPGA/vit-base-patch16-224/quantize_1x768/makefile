#
# Per-case Make rules for FPGA operator 'quantize_1x768' (model
# 'vit-base-patch16-224').
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
BOARD_LINK_SCRIPT := $(CURDIR)/rodata-shifted.ld
include ../common.mk
include ../board.mk
