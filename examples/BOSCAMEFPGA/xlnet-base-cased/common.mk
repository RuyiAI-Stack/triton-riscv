# Each case owns its kernel/configuration; Make owns every compiler stage.
.DEFAULT_GOAL := asm
.DELETE_ON_ERROR:
CASE := $(notdir $(CURDIR))
VIT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(VIT_ROOT)/../../..)
BUILD_FILES := $(VIT_ROOT)/common.mk $(CURDIR)/makefile
PYTHON ?= python3
KERNELS ?= $(VIT_ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
OUT := $(KERNELS)/$(CASE)
REPORTER := $(VIT_ROOT)/tools/report.py
EXPORTER := $(VIT_ROOT)/tools/export.py
FRONTEND := $(REPO_ROOT)/examples/BOSCAMEFPGA/tools/contract_frontend.py
FRONTEND_ID := $(KERNELS)/.frontend-inputs.json
BUDDY_ROOT ?= $(REPO_ROOT)/buddy-mlir
BUDDY_BIN ?= $(or $(BUDDY_MLIR_BINARY_DIR),$(BUDDY_ROOT)/build/bin)
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(BUDDY_ROOT)/llvm/build/bin)
BUDDY_OPT ?= $(BUDDY_BIN)/buddy-opt
BUDDY_TRANSLATE ?= $(BUDDY_BIN)/buddy-translate
LLC ?= $(LLVM_BIN)/llc
CLANG ?= $(LLVM_BIN)/clang
BUDDY_TOOLS := $(BUDDY_ROOT)/examples/FPGA-BOSCAME/tools
export BUDDY_ROOT BUDDY_OPT BUDDY_TRANSLATE LLVM_BIN
export PYTHONPATH := $(VIT_ROOT)/tools:$(PYTHONPATH)

LLC_FLAGS ?= -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d \
  -mattr=+m,+a,+f,+d,+c,+v,+xboscame -code-model=medium
ENCODE_FLAGS ?= --gpr-mode=fixed --preserve-matrix-registers
RESTRICT_FLAGS ?= --expand-f32-sub --expand-vector-merges --order-vector-copies
CLANG_FLAGS ?= --target=riscv64-unknown-elf -march=rv64imafdcv_zicsr_zifencei -mabi=lp64d -mno-relax
OPT_FILE := $(shell command -v "$(BUDDY_OPT)" 2>/dev/null)
TRANSLATE_FILE := $(shell command -v "$(BUDDY_TRANSLATE)" 2>/dev/null)
LLC_FILE := $(shell command -v "$(LLC)" 2>/dev/null)
CLANG_FILE := $(shell command -v "$(CLANG)" 2>/dev/null)

.PHONY: all ttir boscame llvm asm bin simcheck clean FORCE
all: asm
FORCE:

$(sort $(KERNELS) $(OUT)):
	mkdir -p "$@"

# The suite checks the installed frontend once before dispatching its cases.
# Standalone invocations perform the same check themselves. The helper only
# changes the identity file when the frontend implementation actually changes.
ifneq ($(FRONTEND_CHECKED),1)
$(FRONTEND_ID): FORCE
endif
$(FRONTEND_ID): | $(KERNELS)
	@$(PYTHON) -B "$(FRONTEND)" --identity "$@" || { rm -f "$(OUT)/build-result.json"; exit 1; }

# Record effective commands/flags so changing an override invalidates cached
# artifacts too. Unchanged inputs preserve this file's timestamp.
$(OUT)/.compiler-inputs.json: FORCE $(REPORTER) | $(OUT)
	@$(PYTHON) -B "$(REPORTER)" inputs --inputs "$@" \
	  --value 'opt=$(BUDDY_OPT)' --value 'translate=$(BUDDY_TRANSLATE)' \
	  --value 'llc=$(LLC) $(LLC_FLAGS)' --value 'clang=$(CLANG) $(CLANG_FLAGS)' \
	  --value 'encode=$(PYTHON) $(BUDDY_TOOLS)/ame_to_word.py $(ENCODE_FLAGS)' \
	  --value 'restrict=$(PYTHON) $(BUDDY_TOOLS)/restrict_fpga_assembly.py $(RESTRICT_FLAGS)'

# Invalidate old PASS evidence before touching an artifact. A failed command
# records its stage/log; successful commands are followed by the public-stage
# report. Redirection filters publish only complete output via an atomic rename.
define run_stage
	@rm -f "$(OUT)/build-result.json"
	@set -eu; \
	  trap 'code=$$?; rm -f "$@.tmp"; if test $$code -ne 0; then cat "$(OUT)/$(1).log" >&2; $(PYTHON) -B "$(REPORTER)" failure --case "$(CASE)" --output "$(KERNELS)" --stage "$(1)" --log "$(OUT)/$(1).log" || :; fi' 0; \
	  printf '%s\n' '$ $(2)' > "$(OUT)/$(1).log"; \
	  { $(2); } >> "$(OUT)/$(1).log" 2>&1
endef

FRONTEND_OUTPUTS := $(addprefix $(OUT)/,kernel.ttir kernel.frontend.mlir contract-manifest.json adapter.c frontend.json)
$(FRONTEND_OUTPUTS) &: $(CURDIR)/kernel.py $(VIT_ROOT)/cases.py $(EXPORTER) $(FRONTEND) $(FRONTEND_ID) $(BUILD_FILES) | $(OUT)
	$(call run_stage,ttir,$(PYTHON) -B "$(EXPORTER)" --case "$(CASE)" --output "$(OUT)")

BACKEND_INPUTS := $(BUILD_FILES) $(OUT)/.compiler-inputs.json
$(OUT)/kernel.bufferized.mlir: $(OUT)/kernel.frontend.mlir $(FRONTEND_OUTPUTS) $(BACKEND_INPUTS) $(OPT_FILE)
	$(call run_stage,bufferized,"$(BUDDY_OPT)" "$<" --kernel-contract-verify -o "$@")
$(OUT)/kernel.planned.mlir: $(OUT)/kernel.bufferized.mlir $(BACKEND_INPUTS) $(OPT_FILE)
	$(call run_stage,planned,"$(BUDDY_OPT)" "$<" --bosc-ame-plan --bosc-ame-plan-verify -o "$@")
$(OUT)/kernel.boscame.mlir: $(OUT)/kernel.planned.mlir $(BACKEND_INPUTS) $(OPT_FILE)
	$(call run_stage,boscame,"$(BUDDY_OPT)" "$<" --bosc-ame-emit -o "$@")
$(OUT)/kernel.llvm.mlir: $(OUT)/kernel.boscame.mlir $(BACKEND_INPUTS) $(OPT_FILE)
	$(call run_stage,llvm,"$(BUDDY_OPT)" "$<" --lower-bosc-ame -o "$@")
$(OUT)/kernel.ll: $(OUT)/kernel.llvm.mlir $(BACKEND_INPUTS) $(TRANSLATE_FILE)
	$(call run_stage,llvm-ir,"$(BUDDY_TRANSLATE)" --buddy-to-llvmir "$<" -o "$@")
$(OUT)/kernel.s: $(OUT)/kernel.ll $(BACKEND_INPUTS) $(LLC_FILE)
	$(call run_stage,asm,"$(LLC)" "$<" $(LLC_FLAGS) -o "$@")
$(OUT)/kernel.encoded.s: $(OUT)/kernel.s $(BACKEND_INPUTS) $(BUDDY_TOOLS)/ame_to_word.py
	$(call run_stage,encode,"$(PYTHON)" "$(BUDDY_TOOLS)/ame_to_word.py" $(ENCODE_FLAGS) < "$<" > "$@.tmp"; mv "$@.tmp" "$@")
$(OUT)/kernel.nr.s: $(OUT)/kernel.encoded.s $(BACKEND_INPUTS) $(BUDDY_TOOLS)/restrict_fpga_assembly.py
	$(call run_stage,restrict,"$(PYTHON)" "$(BUDDY_TOOLS)/restrict_fpga_assembly.py" $(RESTRICT_FLAGS) < "$<" > "$@.tmp"; mv "$@.tmp" "$@")
$(OUT)/kernel.o: $(OUT)/kernel.nr.s $(BACKEND_INPUTS) $(CLANG_FILE)
	$(call run_stage,object,"$(CLANG)" $(CLANG_FLAGS) -c "$<" -o "$@")

ttir: $(FRONTEND_OUTPUTS)
boscame: $(OUT)/kernel.boscame.mlir $(FRONTEND_OUTPUTS)
llvm: $(OUT)/kernel.ll $(FRONTEND_OUTPUTS)
asm: $(OUT)/kernel.o $(FRONTEND_OUTPUTS)
ttir boscame llvm asm:
	@$(PYTHON) -B "$(REPORTER)" success --case "$(CASE)" --output "$(KERNELS)" --stage "$@" $(if $(REPORT),--report "$(REPORT)")
simcheck: boscame
	$(PYTHON) -B "$(VIT_ROOT)/tools/simcheck.py" --case "$(CASE)" --output "$(KERNELS)" $(if $(COLLECT),--collect,)
clean:
	rm -rf "$(OUT)"

# asm compiles kernel.o. board.mk links that object into the NH/E6 image.
bin: asm
	$(MAKE) -f $(VIT_ROOT)/board.mk bin
