#
# Per-case Make rules for FPGA operator 'makefile' (model 'xlnet-base-cased').
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

.PHONY: all ttir boscame llvm asm _suite $(CASE_TARGETS) FORCE simcheck launchcheck hostcheck modelcheck assets weights bin clean
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
	$(PYTHON) -B "$(ROOT)/tools/simcheck.py" $(CASE_ARGS) --output "$(KERNELS)"
launchcheck:
	$(PYTHON) -B "$(ROOT)/tools/check_launch.py" $(CASE_ARGS) --build-dir "$(KERNELS)"
hostcheck:
	TRITON_INTERPRET=1 $(PYTHON) -B "$(ROOT)/tools/hostcheck.py" $(CASE_ARGS)
modelcheck:
	$(PYTHON) -B "$(ROOT)/tools/check_model.py"
	$(PYTHON) -B "$(ROOT)/tools/check_model_abi.py"
assets:
	$(PYTHON) -B "$(ROOT)/tools/fetch_assets.py"
weights:
	$(PYTHON) -B "$(ROOT)/tools/prepare_weights.py" --assets "$(ASSETS)" $(WEIGHT_FLAGS)
bin: asm
	+@code=0; for name in $(CASES); do $(MAKE) --no-print-directory -C "$(ROOT)/$$name" bin KERNELS="$(KERNELS)" PYTHON="$(PYTHON)" || code=1; done; exit $$code
clean:
	rm -rf "$(ROOT)/build"

# Compile the actual 12-layer C orchestration; model weights/linking are separate.
CLANG ?= $(BUDDY_ROOT)/llvm/build/bin/clang
.PHONY: model-runtime-object
model-runtime-object:
	mkdir -p "$(ROOT)/build/model_runtime"
	$(CLANG) --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei -mabi=lp64d \
	  -mcmodel=medany -mno-relax -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize \
	  -I"$(ROOT)" -I"$(ROOT)/model_runtime" -c "$(ROOT)/model_runtime/model.c" \
	  -o "$(ROOT)/build/model_runtime/model.o"

.PHONY: bincheck
bincheck:
	$(PYTHON) -B "$(ROOT)/tools/check_bins.py" --kernels "$(KERNELS)" \
	  --objcopy "$(BUDDY_ROOT)/llvm/build/bin/llvm-objcopy"

.PHONY: model-input model-reference model-bin model-host
model-input:
	$(PYTHON) -B "$(ROOT)/tools/prepare_model_input.py" --assets "$(ASSETS)" $(MODEL_INPUT_FLAGS)
model-reference:
	$(PYTHON) -B "$(ROOT)/tools/quantized_model_reference.py" $(MODEL_REFERENCE_FLAGS)
model-bin:
	$(PYTHON) -B "$(ROOT)/tools/build_model_bin.py" --kernels "$(KERNELS)" $(MODEL_BUILD_FLAGS)
model-host:
	$(PYTHON) -B "$(ROOT)/tools/build_model_host.py" --kernels "$(KERNELS)" $(MODEL_HOST_FLAGS)
