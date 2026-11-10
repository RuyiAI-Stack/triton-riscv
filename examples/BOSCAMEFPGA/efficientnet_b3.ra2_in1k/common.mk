#
# Shared Make rules for BOSCAME FPGA model 'efficientnet_b3.ra2_in1k'.
#
# Inputs: case KERNEL_CONFIG, Triton kernel.py, buddy-opt, and the RISC-V GNU
# toolchain plus board flags.
#
# Outputs: TTIR/MLIR dumps, objects, ELF, and .bin images under the model build
# directory.
#
# When used: included by each case makefile and the model top-level makefile for
# AOT and board builds.
#
.DEFAULT_GOAL := asm
.DELETE_ON_ERROR:
CASE := $(notdir $(CURDIR))
VIT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(VIT_ROOT)/../../..)
BUILD_FILES := $(VIT_ROOT)/common.mk $(CURDIR)/makefile \
  $(VIT_ROOT)/quantized_conv_kernels.py \
  $(VIT_ROOT)/tools/quantized_conv_pipeline.py \
  $(VIT_ROOT)/diagnostics/model_integration/silu_rvv_direct_output_candidate.c \
  $(VIT_ROOT)/tools/build_silu_native.py
ifneq ($(filter gap_%,$(CASE)),)
BUILD_FILES += $(VIT_ROOT)/diagnostics/model_integration/gap_torch_order_candidate.c \
  $(VIT_ROOT)/tools/build_gap_native.py
endif
ifeq ($(CASE),linear_1x1000x1536)
BUILD_FILES += $(VIT_ROOT)/quantized_linear_kernels.py \
  $(VIT_ROOT)/tools/quantized_linear_pipeline.py \
  $(VIT_ROOT)/matmul_1x1000x1536/kernel.py
endif
PYTHON ?= python3
KERNELS ?= $(VIT_ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
OUT := $(KERNELS)/$(CASE)
REPORTER := $(VIT_ROOT)/tools/report.py
EXPORTER := $(VIT_ROOT)/tools/export.py
FRONTEND := $(REPO_ROOT)/examples/BOSCAMEFPGA/tools/contract_frontend.py
FRONTEND_ID := $(KERNELS)/.frontend-inputs.json
BUDDY_ROOT ?= $(or $(BUDDY_DIR),$(REPO_ROOT)/buddy-mlir)
BUDDY_BIN ?= $(or $(BUDDY_MLIR_BINARY_DIR),$(BUDDY_ROOT)/build/bin)
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(BUDDY_ROOT)/llvm/build/bin)
BUDDY_OPT ?= $(BUDDY_BIN)/buddy-opt
BUDDY_TRANSLATE ?= $(BUDDY_BIN)/buddy-translate
LLC ?= $(LLVM_BIN)/llc
LLVM_OPT ?= $(LLVM_BIN)/opt
CLANG ?= $(LLVM_BIN)/clang
BUDDY_TOOLS := $(BUDDY_ROOT)/examples/FPGA-BOSCAME/tools
include $(BUDDY_TOOLS)/ordered_reduction_policy.mk
BUILD_FILES += $(FPGA_REDUCTION_POLICY_MAKEFILE)
export BUDDY_ROOT BUDDY_OPT BUDDY_TRANSLATE LLVM_BIN
export PYTHONPATH := $(VIT_ROOT)/tools:$(PYTHONPATH)

LLC_FLAGS ?= -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d \
  -mattr=+m,+a,+f,+d,+c,+v,+xboscame -code-model=medium
ENCODE_FLAGS ?= --gpr-mode=fixed --preserve-matrix-registers
RESTRICT_FLAGS ?= --expand-f32-sub --expand-vector-merges --order-vector-copies
CLANG_FLAGS ?= --target=riscv64-unknown-elf -march=rv64imafdcv_zicsr_zifencei -mabi=lp64d -mno-relax
# Opt in only from the 18 staged W8A8 convolution case makefiles. Preserve
# strict FP operations and keep LLVM's loop/SLP vectorizers disabled.
CONV_LLVM_SCALAR_O2 ?= 0
# Only staged W8A8 convolutions may select this raw-im2col coherence step.
# It preserves the LLVM math and does not touch the INT8 MMA stages.
CONV_PATCH_DDR_PHASES ?= 0
CONV_LLVM_OPT_INPUT := $(OUT)/kernel.ll
ifeq ($(CONV_PATCH_DDR_PHASES),1)
ifneq ($(CONV_LLVM_SCALAR_O2),1)
$(error CONV_PATCH_DDR_PHASES requires CONV_LLVM_SCALAR_O2=1)
endif
CONV_PHASE_TOOL := $(VIT_ROOT)/tools/add_conv_patch_ddr_phases.py
BUILD_FILES += $(CONV_PHASE_TOOL)
CONV_LLVM_OPT_INPUT := $(OUT)/kernel.phase.ll
CONV_PHASE_REPORT := $(OUT)/kernel.phase.json
CONV_PHASE_INPUTS := --value 'conv-patch-ddr-phases=$(CONV_PHASE_TOOL)'
endif
# Compilation is independent of board history. Optionally attach exact proof.
GAP_NUMERICAL_EVIDENCE ?=
ifneq ($(filter gap_%,$(CASE)),)
GAP_NUMERICAL_INPUTS := --value 'gap-numerical-evidence=$(GAP_NUMERICAL_EVIDENCE)'
BUILD_FILES += $(GAP_NUMERICAL_EVIDENCE)
endif
ifeq ($(CONV_LLVM_SCALAR_O2),1)
LLVM_SCALAR_O2_FLAGS := -O2 -S -vectorize-loops=false -vectorize-slp=false
LLVM_OPT_FILE := $(shell command -v "$(LLVM_OPT)" 2>/dev/null)
LLVM_ASM_INPUT := $(OUT)/kernel.optimized.ll
LLVM_SCALAR_O2_INPUTS := --value 'llvm-scalar-o2-tool=$(LLVM_OPT)' \
  --value 'llvm-scalar-o2-flags=$(LLVM_SCALAR_O2_FLAGS)'
else
LLVM_ASM_INPUT := $(OUT)/kernel.ll
endif
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
	  --value 'opt=$(BUDDY_OPT)' --value 'reduction-lowering=$(FPGA_REDUCTION_LOWER_OPTIONS)' --value 'translate=$(BUDDY_TRANSLATE)' \
	  --value 'llc=$(LLC) $(LLC_FLAGS)' --value 'clang=$(CLANG) $(CLANG_FLAGS)' \
	  --value 'encode=$(PYTHON) $(BUDDY_TOOLS)/ame_to_word.py $(ENCODE_FLAGS)' \
	  --value 'restrict=$(PYTHON) $(BUDDY_TOOLS)/restrict_fpga_assembly.py $(RESTRICT_FLAGS) $(FPGA_REDUCTION_FILTER_FLAGS)' \
	  $(LLVM_SCALAR_O2_INPUTS) $(GAP_NUMERICAL_INPUTS) $(CONV_PHASE_INPUTS)

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
$(OUT)/kernel.llvm.mlir: $(OUT)/kernel.boscame.mlir $(BACKEND_INPUTS) $(OPT_FILE) $(OUT)/.ordered-reduction-policy
	$(call run_stage,llvm,"$(BUDDY_OPT)" "$<" --lower-bosc-ame="$(FPGA_REDUCTION_LOWER_OPTIONS)" -o "$@")
$(OUT)/kernel.ll: $(OUT)/kernel.llvm.mlir $(BACKEND_INPUTS) $(TRANSLATE_FILE)
	$(call run_stage,llvm-ir,"$(BUDDY_TRANSLATE)" --buddy-to-llvmir "$<" -o "$@")
ifeq ($(CONV_PATCH_DDR_PHASES),1)
$(OUT)/kernel.phase.ll $(OUT)/kernel.phase.json &: $(OUT)/kernel.ll $(BACKEND_INPUTS) $(CONV_PHASE_TOOL)
	$(call run_stage,llvm-phase,$(PYTHON) -B "$(CONV_PHASE_TOOL)" --case "$(CASE)" --input "$(OUT)/kernel.ll" --output "$(OUT)/kernel.phase.ll" --report "$(OUT)/kernel.phase.json")
endif
ifeq ($(CONV_LLVM_SCALAR_O2),1)
$(OUT)/kernel.optimized.ll: $(CONV_LLVM_OPT_INPUT) $(CONV_PHASE_REPORT) $(BACKEND_INPUTS) $(LLVM_OPT_FILE)
	$(call run_stage,llvm-opt,"$(LLVM_OPT)" $(LLVM_SCALAR_O2_FLAGS) "$<" -o "$@")
endif
ifneq ($(filter silu_%,$(CASE)),)
$(OUT)/kernel.s: $(OUT)/kernel.ll $(BACKEND_INPUTS) $(CLANG_FILE) $(VIT_ROOT)/diagnostics/model_integration/silu_rvv_direct_output_candidate.c
	$(call run_stage,native-rvv,"$(CLANG)" $(CLANG_FLAGS) -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -ffp-contract=off -Defficientnet_silu_rvv_direct_output_candidate=efficientnet_rvv_$(CASE) -S "$(VIT_ROOT)/diagnostics/model_integration/silu_rvv_direct_output_candidate.c" -o "$@")
else ifneq ($(filter gap_%,$(CASE)),)
$(OUT)/kernel.s: $(OUT)/kernel.ll $(BACKEND_INPUTS) $(CLANG_FILE) $(VIT_ROOT)/diagnostics/model_integration/gap_torch_order_candidate.c
	$(call run_stage,native-ordered,"$(CLANG)" $(CLANG_FLAGS) -mcmodel=medany -O2 -ffreestanding -fno-builtin -fno-vectorize -fno-slp-vectorize -ffp-contract=off -Defficientnet_gap_torch_order_candidate=efficientnet_phase_$(CASE) -S "$(VIT_ROOT)/diagnostics/model_integration/gap_torch_order_candidate.c" -o "$@")
else
$(OUT)/kernel.s: $(LLVM_ASM_INPUT) $(BACKEND_INPUTS) $(LLC_FILE)
	$(call run_stage,asm,"$(LLC)" "$<" $(LLC_FLAGS) -o "$@")
endif
$(OUT)/kernel.encoded.s: $(OUT)/kernel.s $(BACKEND_INPUTS) $(BUDDY_TOOLS)/ame_to_word.py
	$(call run_stage,encode,"$(PYTHON)" "$(BUDDY_TOOLS)/ame_to_word.py" $(ENCODE_FLAGS) < "$<" > "$@.tmp"; mv "$@.tmp" "$@")
$(OUT)/kernel.nr.s: $(OUT)/kernel.encoded.s $(BACKEND_INPUTS) $(BUDDY_TOOLS)/restrict_fpga_assembly.py
	$(call run_stage,restrict,"$(PYTHON)" "$(BUDDY_TOOLS)/restrict_fpga_assembly.py" $(RESTRICT_FLAGS) $(FPGA_REDUCTION_FILTER_FLAGS) < "$<" > "$@.tmp"; mv "$@.tmp" "$@")
$(OUT)/kernel.o: $(OUT)/kernel.nr.s $(BACKEND_INPUTS) $(CLANG_FILE)
	$(call run_stage,object,"$(CLANG)" $(CLANG_FLAGS) -c "$<" -o "$@")
ifneq ($(filter silu_%,$(CASE)),)
	@$(PYTHON) -B "$(VIT_ROOT)/tools/build_silu_native.py" --case "$(CASE)" --output "$(OUT)" --clang "$(CLANG)" --tool-root "$(BUDDY_ROOT)"
endif
ifneq ($(filter gap_%,$(CASE)),)
	@$(PYTHON) -B "$(VIT_ROOT)/tools/build_gap_native.py" --case "$(CASE)" --output "$(OUT)" --clang "$(CLANG)" --tool-root "$(BUDDY_ROOT)" $(if $(GAP_NUMERICAL_EVIDENCE),--numerical-evidence "$(GAP_NUMERICAL_EVIDENCE)")
endif

ttir: $(FRONTEND_OUTPUTS)
boscame: $(OUT)/kernel.boscame.mlir $(FRONTEND_OUTPUTS)
llvm: $(LLVM_ASM_INPUT) $(FRONTEND_OUTPUTS)
asm: $(OUT)/kernel.o $(FRONTEND_OUTPUTS)
ttir boscame llvm asm:
	@$(PYTHON) -B "$(REPORTER)" success --case "$(CASE)" --output "$(KERNELS)" --stage "$@" $(if $(REPORT),--report "$(REPORT)")
simcheck: boscame
	$(PYTHON) -B "$(VIT_ROOT)/tools/simcheck.py" --case "$(CASE)" --output "$(KERNELS)" $(if $(COLLECT),--collect,)
clean:
	rm -rf "$(OUT)"

# asm compiles kernel.o. board.mk links that object into the NH/E6 image.
bin: asm
	+$(MAKE) -f "$(VIT_ROOT)/board.mk" bin KERNELS="$(KERNELS)" PYTHON="$(PYTHON)"
