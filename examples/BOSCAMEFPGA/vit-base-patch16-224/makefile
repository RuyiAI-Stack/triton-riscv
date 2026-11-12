#
# Per-case Make rules for FPGA operator 'makefile' (model
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
.DEFAULT_GOAL := all
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
ASSETS ?= $(ROOT)/assets
KERNELS ?= $(ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
IMPORT ?= $(ROOT)/build/import
MODE ?= w8a8
ifneq ($(MODE),w8a8)
$(error Only MODE=w8a8 is supported for FPGA; use the independent host FP32 baseline for accuracy)
endif
REPLACEMENT ?= $(ROOT)/build/replacement-$(MODE)
BUDDY_ROOT ?= $(abspath $(ROOT)/../../../buddy-mlir)
export BUDDY_ROOT
export PYTHONPATH := $(ROOT)/tools:$(PYTHONPATH)
ALL_CASES := $(notdir $(patsubst %/kernel.py,%,$(wildcard $(ROOT)/*/kernel.py)))
# An explicit CASES selection writes a subset report, preserving suite evidence.
SCOPE := $(if $(filter undefined,$(origin CASES)),--all,)
CASES ?= $(ALL_CASES)
CASE_ARGS = $(foreach case,$(sort $(CASES)),--case $(case))
CASE_TARGETS := $(addprefix build-,$(sort $(CASES)))
REPORTER := $(ROOT)/tools/report.py
FRONTEND := $(abspath $(ROOT)/../tools/contract_frontend.py)
FRONTEND_ID := $(KERNELS)/.frontend-inputs.json
STAGE ?= asm

.PHONY: all ttir boscame llvm asm _suite $(CASE_TARGETS) FORCE simcheck w8a8-chaincheck patch-chaincheck operatorcheck model-simcheck launchcheck assets import replace lower model-bin clean
all: asm
ttir boscame llvm asm:
	+@code=0; $(MAKE) --no-print-directory -k _suite STAGE=$@ || code=$$?; \
	  $(PYTHON) -B "$(REPORTER)" collect --output "$(KERNELS)" --stage "$@" $(CASE_ARGS) $(SCOPE) $(if $(REPORT),--report "$(REPORT)") || code=1; \
	  exit $$code

# -k attempts every selected case, including when another case fails. Recursive
# Make shares the parent's jobserver, so make -jN also builds cases in parallel.
_suite: $(CASE_TARGETS)
$(CASE_TARGETS): build-%: $(FRONTEND_ID)
	+@rm -f "$(KERNELS)/$*/build-result.json"; \
	  $(MAKE) --no-print-directory -C "$(ROOT)/$*" $(STAGE) KERNELS="$(KERNELS)" PYTHON="$(PYTHON)" FRONTEND_CHECKED=1 REPORT= || { \
	    if ! test -f "$(KERNELS)/$*/build-result.json"; then \
	      $(PYTHON) -B "$(REPORTER)" failure --case "$*" --output "$(KERNELS)" --stage "$(STAGE)" || :; \
	    fi; exit 1; }
FORCE:
$(KERNELS):
	mkdir -p "$@"
$(FRONTEND_ID): FORCE | $(KERNELS)
	@$(PYTHON) -B "$(FRONTEND)" --identity "$@" > "$(KERNELS)/frontend-identity.log" 2>&1 || { \
	  cat "$(KERNELS)/frontend-identity.log" >&2; \
	  for name in $(CASES); do \
	    $(PYTHON) -B "$(REPORTER)" failure --case "$$name" --output "$(KERNELS)" --stage ttir --log "$(KERNELS)/frontend-identity.log" || :; \
	  done; exit 1; }

simcheck:
	$(PYTHON) -B "$(ROOT)/tools/simcheck.py" $(if $(SCOPE),--all,$(CASE_ARGS)) --output "$(KERNELS)" $(if $(COLLECT),--collect,)
patch-chaincheck:
	$(PYTHON) -B "$(ROOT)/tools/check_patch_chain.py" --kernels "$(KERNELS)" $(if $(COLLECT),--collect,) $(if $(REPORT),--report "$(REPORT)")
operatorcheck:
	$(PYTHON) -B "$(ROOT)/tools/check_all_operators.py" --kernels "$(KERNELS)" $(if $(REPORT_DIR),--report-dir "$(REPORT_DIR)")
w8a8-chaincheck:
	+$(MAKE) boscame CASES='quantize_1x768 quantize_1000x768 matmul_i8_1x1000x768 dequantize_1x1000 bias_add_1x1000 add_1x1000'
	$(PYTHON) -B "$(ROOT)/tools/check_w8a8_chain.py" --kernels "$(KERNELS)"
model-simcheck:
	$(PYTHON) -B "$(ROOT)/tools/model_simcheck.py" --kernels "$(KERNELS)" --mode $(MODE) --replacement "$(REPLACEMENT)" --checkpoint "$(ASSETS)/model.safetensors"
launchcheck:
	$(PYTHON) -B "$(ROOT)/tools/check_launch.py" $(if $(SCOPE),--all,$(CASE_ARGS)) --build-dir "$(KERNELS)"
assets:
	$(PYTHON) -B "$(ROOT)/tools/fetch_assets.py" --output "$(ASSETS)"
import:
	$(PYTHON) -B "$(ROOT)/tools/import_model.py" --assets "$(ASSETS)" --checkpoint "$(ASSETS)/model.safetensors" --output "$(IMPORT)" --save-mlir
replace:
	$(PYTHON) -B "$(ROOT)/tools/replace_triton.py" --import-dir "$(IMPORT)" --triton-build "$(KERNELS)" --output "$(REPLACEMENT)" --require-complete --w8a8
lower:
	$(PYTHON) -B "$(ROOT)/tools/lower_model.py" --replacement "$(REPLACEMENT)" --output "$(ROOT)/build/lowered"
clean:
	rm -rf "$(ROOT)/build"
# Whole-model ViT-B/16 image (artifact only; board fit/run tracked separately).
MODEL_OUT ?= $(ROOT)/build/model-bin
model-bin:
	$(PYTHON) -B "$(ROOT)/tools/build_model_bin.py" --output "$(MODEL_OUT)" \
	  --replacement "$(REPLACEMENT)" --kernels "$(KERNELS)" --assets "$(ASSETS)" \
	  $(if $(REUSE_LOWERED),--reuse-lowered,)
