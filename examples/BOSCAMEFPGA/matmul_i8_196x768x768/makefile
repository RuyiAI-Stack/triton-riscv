# makefile -- the matmul_i8_196x768x768 experimental BOSCAME pipeline.
#
#     kernels.py (Triton)                      [make ttir]
#       -> kernel.ttir
#     external export_contract(ASTSource, target=boscame)
#       -> kernel.frontend.mlir (neutral buffer contract)
#       -> kernel.bufferized.mlir --bosc-ame-plan
#       -> kernel.planned.mlir --bosc-ame-emit             [make boscame]
#       -> kernel.boscame.mlir
#     --lower-bosc-ame + --buddy-to-llvmir     [make llvm]
#       -> kernel.llvm.mlir + kernel.ll
#     llc -> AME fixed-register/fence post     [make asm]
#       -> kernel.nr.s
#     clang -c + ld.lld + audit + objcopy      [make bin]
#       -> build/matmul_i8_196x768x768/matmul_i8_196x768x768.bin
#
# `make` runs all of it.  Every stage is a file rule, so re-running only
# rebuilds what changed.
#
# The problem is A[196,768] i8 @ B[768,768] i8^T + C[196,768] i32 with the
# BOSCAME tile contract BM=32, BN=BK=128.  The AME register file caps one tile
# dimension at 64, so the planner lowers BN=128 as two 64-column AME groups and
# BK=128 as two 64-wide AME K blocks.  M=196 is not a multiple of BM=32, so the
# launch grid is 7x6 and the last M program is the masked boundary tile: the
# frontend uses the explicit launch grid to specialize full/tail programs,
# retains the K/BK reduction loop, and accesses only the logical [196,768] rows.
#
# The image runs on the NH/E6 platform (runtime/boscame/nh_e6), which owns the
# boot, the dual-core release/wait, the E6 address-space setup (seven TOR PMP
# regions, `satp = 0`) and the CSR 0xbc0 cache policy.  `link.ld` places the
# DDR operand windows inside the PMP region the platform exports -- otherwise a
# DMA that leaves that region hangs with no diagnostic.
#
# Board-check and debug variants build into their own directories:
#     make debug          progress marker per program + a 20-row on-board oracle
#     make boardcheck     two full launches, signed-i8 extremes, INT32 wrap,
#                         input/DDR guards and the full 196-row oracle
# `make FENCES=1` restores the board `fence rw, rw` pairs around AME/RVV
# memory instructions and the strict ELF fence audit.
.DEFAULT_GOAL := all
.DELETE_ON_ERROR :=

ROOT := $(patsubst %/,%,$(abspath $(dir $(lastword $(MAKEFILE_LIST)))))
BUILD_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
# Export-only goals do not need Buddy. Image/simulator goals use an explicit
# external checkout for its runtime, assembly filters and functional model.
BUDDY_ROOT ?=
export BUDDY_ROOT
ifneq ($(filter-out ttir frontend config clean,$(or $(MAKECMDGOALS),all)),)
ifeq ($(strip $(BUDDY_ROOT)),)
$(error set BUDDY_ROOT to the external Buddy checkout for backend goals)
endif
endif
REPO := $(if $(BUDDY_ROOT),$(abspath $(BUDDY_ROOT)))
FRONTEND_TOOLS := $(abspath $(ROOT)/../tools)
TOOLS := $(REPO)/examples/FPGA-BOSCAME/tools

# Shared platform toolchain: prefers the in-tree LLVM build, then PATH.
REPO_ROOT := $(REPO)
ifneq ($(strip $(BUDDY_ROOT)),)
include $(REPO)/runtime/boscame/toolchain.mk
endif

BUDDY_BIN ?= $(BUDDY_MLIR_BINARY_DIR)
BUDDY_OPT ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-opt,buddy-opt)
BUDDY_TRANSLATE ?= $(if $(BUDDY_BIN),$(BUDDY_BIN)/buddy-translate,buddy-translate)
# Python simulator helpers use the same explicitly selected backend tools.
export BUDDY_OPT BUDDY_TRANSLATE
LLC ?= $(call fpga_llvm_tool,llc)
OBJCOPY ?= $(call fpga_llvm_tool,llvm-objcopy)
OBJDUMP ?= $(call fpga_llvm_tool,llvm-objdump)
NM ?= $(call fpga_llvm_tool,llvm-nm)
# ld.lld ships with LLVM, but a BUDDY build does not require the lld project;
# fall back to the linker on PATH when the in-tree build has none.
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
PYTHON ?= python3

NAME := matmul_i8_196x768x768
OUT := $(ROOT)/build/$(NAME)
# Generated ABI bridge and C shape header (intermediates, not shipped).
SCRATCH := $(ROOT)/build/.scratch
CONFIG := $(SCRATCH)/matmul_i8_config.h
RUNTIME := $(REPO)/runtime/boscame/runtime
UART := $(REPO)/runtime/boscame/uart
PLATFORM := $(REPO)/runtime/boscame/nh_e6
NR := $(REPO)/runtime/boscame/nr

# --- fence policy: board fences around AME/RVV memory instructions -------
# The assembly filter normally inserts a `fence rw, rw` pair around every AME
# and vector memory instruction. This directory defaults to the raw instruction
# stream so the generated kernel.nr.s can be inspected without them; the ELF
# audit is relaxed accordingly. Build with `make FENCES=1` for the fenced image.
FENCES ?= 0
ifeq ($(filter 0 1,$(FENCES)),)
$(error FENCES must be 0 or 1, got '$(FENCES)')
endif
FENCE_FLAGS := $(if $(filter 1,$(FENCES)),,--no-fences)
AUDIT_FENCE_FLAGS := $(if $(filter 1,$(FENCES)),,--allow-unfenced)

# --- build variant (empty for the default production image) -------------
# `VAR` selects a directory and a file suffix so every variant keeps its own
# IR, objects, image and audit:
#   make                -> build/<name>/<name>.bin
#   make debug          -> build/<name>/debug/<name>.debug.bin
#   make boardcheck     -> build/<name>/boardcheck/<name>.boardcheck.bin
VAR ?=
VARIANT := $(VAR)
V_DIR := $(OUT)$(if $(VARIANT),/$(VARIANT))
V_NAME := $(NAME)$(if $(VARIANT),.$(VARIANT),)
V_CFLAGS := $(if $(filter debug,$(VAR)),-DMATMUL_I8_TRACE=1 -DMATMUL_I8_CHECK_ROWS=16) \
  $(if $(filter boardcheck,$(VAR)),-DMATMUL_I8_BOARD_CHECK=1 -DMATMUL_I8_TRACE=1 -DMATMUL_I8_CHECK_ROWS=196)
# The frontend records target-independent dot shapes and caller ABI facts.
# The planner derives AME tiles from those shapes; kernels.py and build.py
# supply the same problem dimensions and launch grid.
# ABI bridge: three typed unranked descriptors + six grid scalars.
ADAPTER := experimental-boscame
ADAPTER_C := $(SCRATCH)/adapter.c
OBJ := $(V_DIR)/obj
ELF := $(V_DIR)/$(V_NAME).elf
BIN := $(V_DIR)/$(V_NAME).bin

RISCV_FLAGS := -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections \
  -I$(ROOT) -I$(RUNTIME) -I$(UART) -I$(PLATFORM) -I$(NR) \
  -I$(SCRATCH)
# Freestanding image: the DMA runtime's programming assertions compile out,
# exactly as in the other FPGA examples; `bosc_ame_dma_wait` keeps its fatal
# hardware checks. DMA moves external DDR tiles into fixed SPM windows; AME
# and one-dimensional RVV transfers connect SPM with their register files.
C_FLAGS := $(RISCV_FLAGS) -fno-vectorize -fno-slp-vectorize '-DBOSC_AME_DMA_ASSERT(x)='
LLC_FLAGS := -O2 -filetype=asm -mtriple=riscv64 -target-abi=lp64d \
  -mattr=+m,+a,+f,+d,+c,+v,+xboscame -code-model=medium

.PHONY: all config ttir boscame llvm asm bin debug boardcheck check audit \
  simcheck clean FORCE
all: bin

config: $(CONFIG)

$(sort $(OUT) $(V_DIR) $(OBJ) $(SCRATCH)):
	mkdir -p $@

# --- 1. Python -> TTIR --------------------------------------------------
# build.py owns the Triton frontend, the fixed problem size and the toolchain
# configuration; the makefile owns the stages and their order.
# Invalidate the handoff when the installed frontend implementation changes.
.PHONY: frontend-force
frontend-force:
$(OUT)/frontend-inputs.json: frontend-force | $(OUT)
	$(PYTHON) -B $(FRONTEND_TOOLS)/contract_frontend.py --identity $@

$(OUT)/kernel.ttir $(OUT)/kernel.frontend.mlir $(OUT)/contract-manifest.json &: $(ROOT)/kernels.py $(ROOT)/build.py $(FRONTEND_TOOLS)/contract_frontend.py $(OUT)/frontend-inputs.json $(BUILD_MAKEFILE) | $(OUT)
	$(PYTHON) -B $(ROOT)/build.py --export-contract
ttir: $(OUT)/kernel.ttir

# --- 2. TTIR -> BOSCAME -------------------------------------------------
# The frontend emits buffer-form Linalg with neutral contract facts. Separate
# plan and emit stages choose DDR/SPM routes and expand the target operations.
# contract.abi records the caller's aligned, nonalias entry arguments.
ifneq ($(V_DIR),$(OUT))
$(V_DIR)/kernel.frontend.mlir: $(OUT)/kernel.frontend.mlir | $(V_DIR)
	cp $< $@
$(V_DIR)/contract-manifest.json: $(OUT)/contract-manifest.json | $(V_DIR)
	cp $< $@
endif

$(V_DIR)/kernel.bufferized.mlir: $(V_DIR)/kernel.frontend.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --kernel-contract-verify -o $@

$(V_DIR)/kernel.planned.mlir: $(V_DIR)/kernel.bufferized.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --bosc-ame-plan --bosc-ame-plan-verify -o $@

$(V_DIR)/kernel.boscame.mlir: $(V_DIR)/kernel.planned.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --bosc-ame-emit -o $@

boscame: $(V_DIR)/kernel.boscame.mlir

# --- 3. BOSCAME -> LLVM dialect -> LLVM IR -----------------------------
# `--lower-bosc-ame` checks the SPM/DMA contract, compiles the bank views to
# fixed addresses, lowers the DMA requests to the runtime ABI, turns the tile
# operations into `bosc_ame.intr.*` with the configuration and byte strides
# the hardware needs, and finishes the memref/func/arith/scf lowering.
$(V_DIR)/kernel.llvm.mlir: $(V_DIR)/kernel.boscame.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_OPT))
	$(BUDDY_OPT) $< --lower-bosc-ame -o $@

$(V_DIR)/kernel.ll: $(V_DIR)/kernel.llvm.mlir $(BUILD_MAKEFILE) $(wildcard $(BUDDY_TRANSLATE))
	$(BUDDY_TRANSLATE) --buddy-to-llvmir $< -o $@

llvm: $(V_DIR)/kernel.ll

# --- 4. RISC-V assembly with the AME instructions -----------------------
$(V_DIR)/kernel.s: $(V_DIR)/kernel.ll $(BUILD_MAKEFILE) $(wildcard $(LLC))
	$(LLC) $< $(LLC_FLAGS) -o $@

# --- 5. NR post-processing: fixed-register words + fences ---------------
$(V_DIR)/kernel.encoded.s: $(V_DIR)/kernel.s $(TOOLS)/ame_to_word.py
	$(PYTHON) $(TOOLS)/ame_to_word.py --gpr-mode=fixed --preserve-matrix-registers < $< > $@

# FORCE keeps the fence policy switch effective without relinking when the
# filtered output is byte-identical (the recipe only replaces the file then).
$(V_DIR)/kernel.nr.s: $(V_DIR)/kernel.encoded.s $(TOOLS)/restrict_fpga_assembly.py FORCE
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) $(TOOLS)/restrict_fpga_assembly.py $(FENCE_FLAGS) --order-vector-copies < $< > "$$tmp"; \
	test -s "$$tmp" || { echo "error: assembly filter produced no output" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

$(V_DIR)/kernel.o: $(V_DIR)/kernel.nr.s $(BUILD_MAKEFILE)
	$(RISCV_CC) $(RISCV_FLAGS) -march=rv64imafdcv_zicsr_zifencei -c $< -o $@

asm: $(V_DIR)/kernel.nr.s

# --- 6. application driver, ABI bridge, platform and DMA runtime ---------
# The MLIR wrapper takes three typed unranked descriptor pointers plus grid
# scalars. The launcher adapter validates the DMA ABI and uses element counts.
# Boot, platform, UART and DMA runtime are linked once.
$(ADAPTER_C): FORCE $(ROOT)/build.py $(BUILD_MAKEFILE) | $(SCRATCH)
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) -B $(ROOT)/build.py --adapter $(ADAPTER) > "$$tmp"; \
	test -s "$$tmp" || { echo "error: adapter generator produced no C source" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

FORCE:

$(CONFIG): FORCE $(ROOT)/build.py $(BUILD_MAKEFILE) | $(SCRATCH)
	@set -eu; \
	tmp="$@.tmp"; \
	trap 'rm -f "$$tmp"' EXIT INT TERM; \
	$(PYTHON) -B $(ROOT)/build.py --config > "$$tmp"; \
	test -s "$$tmp" || { echo "error: shape config generator produced no header" >&2; exit 1; }; \
	if test -r "$@" && cmp -s "$$tmp" "$@"; then rm -f "$$tmp"; else mv -f "$$tmp" "$@"; fi; \
	trap - EXIT INT TERM

$(OBJ)/adapter.o: $(ADAPTER_C) $(CONFIG) $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) $(V_CFLAGS) -c $< -o $@

$(OBJ)/launch.o: $(ROOT)/launch.c $(CONFIG) $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) $(V_CFLAGS) -c $< -o $@

$(OBJ)/bosc_ame_dma.o: $(RUNTIME)/bosc_ame_dma.c $(RUNTIME)/bosc_ame_dma.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

$(OBJ)/nh_e6_boot.o: $(PLATFORM)/nh_e6_boot.S $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(RISCV_FLAGS) -c $< -o $@

$(OBJ)/nh_e6_platform.o: $(PLATFORM)/nh_e6_platform.c $(PLATFORM)/nh_e6_platform.h \
  $(RUNTIME)/bosc_ame_dma.h $(UART)/uart.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

$(OBJ)/uart.o: $(UART)/uart.c $(UART)/uart.h $(BUILD_MAKEFILE) | $(OBJ)
	$(RISCV_CC) $(C_FLAGS) -c $< -o $@

IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o \
  $(OBJ)/uart.o $(OBJ)/bosc_ame_dma.o $(OBJ)/launch.o \
  $(OBJ)/adapter.o $(V_DIR)/kernel.o

# --- 7. NH/E6 image, its audit and the flat .bin ------------------------
$(ELF): $(IMAGE_OBJECTS) $(ROOT)/link.ld $(BUILD_MAKEFILE)
	$(LINKER) -m elf64lriscv --gc-sections -T $(ROOT)/link.ld \
	  -Map=$(V_DIR)/$(V_NAME).map -o $@ $(IMAGE_OBJECTS)

# The DDR operand windows must sit inside the PMP region the platform exports,
# and the platform's PMP/satp sequence must be in the image: dropping it hangs
# on the first AME/DMA access instead of failing visibly.
audit: $(ELF)
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_start" { print $$1 }')" \
	  = "0000000080100000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_start = 0x80100000"; exit 1; }
	@test "$$($(NM) $< | awk '$$3 == "__nh_e6_ddr_window_end" { print $$1 }')" \
	  = "00000000802e0000" \
	  || { echo "FAIL: link.ld must export __nh_e6_ddr_window_end = 0x802e0000"; exit 1; }
	@$(OBJDUMP) -d $< \
	  | grep -oE 'csrw[[:space:]]+(pmpaddr[0-6]|pmpcfg0|satp)|sfence[.]vma' \
	  | tr -s '[:space:]' ' ' \
	  | grep -q 'csrw pmpaddr0 csrw pmpaddr1 csrw pmpaddr2 csrw pmpaddr3 csrw pmpaddr4 csrw pmpaddr5 csrw pmpaddr6 csrw pmpcfg0 csrw satp sfence.vma' \
	  || { echo "FAIL: the platform's PMP/satp setup sequence is missing"; exit 1; }
	@echo "ok: DDR window exported and the PMP/satp setup sequence is in the image"

$(BIN): $(ELF) $(TOOLS)/check_nr_elf.py
	$(PYTHON) $(TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) $(AUDIT_FENCE_FLAGS) --output $(V_DIR)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	@echo "built $@"

bin: $(BIN) audit

# --- 8. build variants --------------------------------------------------
# Progress markers per program and the 16-row prefix + final logical rows
# oracle.
debug:
	$(MAKE) VAR=debug bin

# Two full launches, signed-i8 extremes, INT32 wrapping, input/DDR guards
# and the full 196-row oracle.  The scalar oracle is 115M multiply-adds per
# round, so use `make debug` when a quick verdict is needed inside the
# capture window; the exhaustive numeric claim is `make simcheck`.
boardcheck:
	$(MAKE) VAR=boardcheck check

# --- 9. functional check that needs no board ----------------------------
# Runs the pass output in tools/boscame_buffer_sim.py (functional DMA/SPM/AME
# model) for every program of the grid and compares all logical outputs with a
# NumPy INT32 oracle.
simcheck: boscame
	$(PYTHON) -B $(ROOT)/simcheck.py --module $(V_DIR)/kernel.boscame.mlir --stress

# --- 10. image checks ---------------------------------------------------
check: bin
	@grep -q '# mqma.b.mm' $(V_DIR)/kernel.nr.s || { echo "FAIL: no mqma"; exit 1; }
	$(PYTHON) -B $(ROOT)/build.py --check-abi $(V_DIR)/kernel.ll
	@python3 -c "import json,sys; a=json.load(open('$(V_DIR)/elf-audit.json')); \
	  ok = a['status'] == 'PASS' and not a['errors'] and \
	       (a['ame_instructions'] > 0 and \
	        (a['ame_store_instructions'] > 0 or a['vector_memory_instructions'] > 0)); \
	  sys.exit(0 if ok else 1)" \
	  || { echo "FAIL: the NR ELF audit did not pass"; exit 1; }
	@echo "ok: $(V_NAME) passed the AME checks and the NR ELF audit"

clean:
	rm -rf $(ROOT)/build
