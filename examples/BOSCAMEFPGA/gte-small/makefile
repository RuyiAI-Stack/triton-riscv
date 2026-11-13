#
# Per-case Make rules for FPGA operator 'makefile' (model 'gte-small').
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
.DEFAULT_GOAL := all
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
KERNELS ?= $(ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
ALL_CASES := $(notdir $(patsubst %/kernel.py,%,$(wildcard $(ROOT)/*/kernel.py)))
CASES ?= $(ALL_CASES)
CASE_ARGS := $(foreach case,$(sort $(CASES)),--case $(case))
BUDDY_ROOT ?= $(abspath $(ROOT)/../../../buddy-mlir)
export BUDDY_ROOT
.PHONY: all ttir boscame llvm asm bin simcheck hostcheck launchcheck runtimecheck assets clean
all: asm
ttir boscame llvm asm bin:
	+@code=0; for case in $(CASES); do \
	  $(MAKE) --no-print-directory -C "$(ROOT)/$$case" $@ KERNELS="$(KERNELS)" PYTHON="$(PYTHON)" || code=1; \
	done; exit $$code
simcheck: boscame
	$(PYTHON) -B "$(ROOT)/tools/simcheck.py" $(CASE_ARGS) --output "$(KERNELS)"
hostcheck:
	$(PYTHON) -B "$(ROOT)/tools/hostcheck.py" $(CASE_ARGS)
launchcheck:
	$(PYTHON) -B "$(ROOT)/tools/hostcheck.py" --abi-only $(CASE_ARGS)
runtimecheck:
	$(PYTHON) -B "$(ROOT)/tools/check_model_runtime.py"
assets:
	$(PYTHON) -B "$(ROOT)/tools/fetch_assets.py" --output "$(ROOT)/assets"
clean:
	rm -rf "$(ROOT)/build"
