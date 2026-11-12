#
# Per-case Make rules for FPGA operator 'makefile' (model 'e5-small-v2').
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
MODEL_HOST ?= $(ROOT)/build/model-host
ALL_CASES := $(notdir $(patsubst %/kernel.py,%,$(wildcard $(ROOT)/*/kernel.py)))
CASES ?= $(ALL_CASES)
CASE_ARGS := $(foreach case,$(sort $(CASES)),--case $(case))
BUDDY_ROOT ?= $(abspath $(ROOT)/../../../buddy-mlir)
export BUDDY_ROOT
.PHONY: all help ttir boscame llvm asm bin model-bin model-host simcheck hostcheck launchcheck modelcheck model-runtime-check assets clean
all: asm
help:
	@printf '%s\n' \
	  'E5-small-v2: per-case kernel.py / launch.c / makefile, matching the ViT layout.' \
	  '  hostcheck / launchcheck       Host numerics and launcher ABI / ABI only' \
	  '  modelcheck / model-runtime-check  Host model contract / C routing ABI' \
	  '  ttir / boscame / llvm / asm   Compile all operators, or CASES="case ..."' \
	  '  bin / simcheck               Link operator images / strict emitted simulation' \
	  '  assets                       Download the pinned checkpoint and tokenizer' \
	  '  model-bin                    Build the complete model image (requires assets)' \
	  '  model-host                   Execute the emitted host reference (after model-bin)' \
	  'See README.md for dependencies and optional scheduler/local-validation commands.'
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
modelcheck:
	$(PYTHON) -B "$(ROOT)/tools/model_reference.py" --self-test
model-runtime-check:
	$(PYTHON) -B "$(ROOT)/tools/check_model_runtime.py"
model-bin:
	+$(MAKE) --no-print-directory -C "$(ROOT)" asm CASES="$(ALL_CASES)" KERNELS="$(KERNELS)" PYTHON="$(PYTHON)"
	$(PYTHON) -B "$(ROOT)/tools/build_model_board.py" --kernels "$(KERNELS)"
model-host:
	$(PYTHON) -B "$(ROOT)/tools/build_model_host.py" --kernels "$(KERNELS)" --output "$(MODEL_HOST)"
assets:
	$(PYTHON) -B "$(ROOT)/tools/fetch_assets.py" --output "$(ROOT)/assets"
clean:
	rm -rf "$(ROOT)/build"
