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
.DEFAULT_GOAL := bin
.DELETE_ON_ERROR:
CASE := $(notdir $(CURDIR))
MODEL := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MODEL)/../../..)
BUDDY_ROOT ?= $(or $(BUDDY_DIR),$(REPO_ROOT)/buddy-mlir)
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(BUDDY_ROOT)/llvm/build/bin)
BUDDY_TOOLS := $(BUDDY_ROOT)/examples/FPGA-BOSCAME/tools
RUNTIME := $(BUDDY_ROOT)/runtime/boscame
PYTHON ?= python3
KERNELS ?= $(MODEL)/build/kernels
KERNELS := $(abspath $(KERNELS))
override OUT := $(KERNELS)/$(CASE)
OBJ := $(OUT)/obj
ELF := $(OUT)/$(CASE).elf
BIN := $(OUT)/$(CASE).bin
REPORTER := $(MODEL)/tools/report.py
BOARD_IDENTITY := $(OBJ)/.board-inputs.json
BOARD_LINK_SCRIPT ?= $(MODEL)/link.ld
BOARD_BUILD_INPUTS := $(MODEL)/board.mk $(CURDIR)/makefile

CLANG ?= $(LLVM_BIN)/clang
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
OBJCOPY ?= $(LLVM_BIN)/llvm-objcopy
OBJDUMP ?= $(LLVM_BIN)/llvm-objdump
CLANG_FILE := $(shell command -v "$(CLANG)" 2>/dev/null)
LINKER_FILE := $(shell command -v "$(LINKER)" 2>/dev/null)
OBJCOPY_FILE := $(shell command -v "$(OBJCOPY)" 2>/dev/null)
OBJDUMP_FILE := $(shell command -v "$(OBJDUMP)" 2>/dev/null)
include $(BUDDY_TOOLS)/ordered_reduction_policy.mk
BOARD_BUILD_INPUTS += $(FPGA_REDUCTION_POLICY_MAKEFILE)

# The image releases a single E6 executor, matching the local NH/E6 runtime.
BOARD_PRIVATE_DDR_NR_SINGLE_E6 ?= 1
ifneq ($(words $(BOARD_PRIVATE_DDR_NR_SINGLE_E6)),1)
$(error BOARD_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1)
endif
ifeq ($(filter 0 1,$(BOARD_PRIVATE_DDR_NR_SINGLE_E6)),)
$(error BOARD_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1)
endif
BOARD_DMA_FLAGS := -DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=$(BOARD_PRIVATE_DDR_NR_SINGLE_E6)
BOARD_FLAGS ?= --target=riscv64-unknown-elf \
  -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -fno-stack-protector -ffunction-sections -fdata-sections -ffp-contract=off \
  -fno-vectorize -fno-slp-vectorize \
  -I$(MODEL) -I$(RUNTIME)/runtime -I$(RUNTIME)/uart \
  -I$(RUNTIME)/nh_e6 -I$(RUNTIME)/nr \
  '-DBOSC_AME_DMA_ASSERT(x)=' -DUART_FREQ_HZ=5529600 \
  -DVIT_WORKSPACE_BYTES=16777216

IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o $(OBJ)/uart.o \
  $(OBJ)/bosc_ame_dma.o $(OBJ)/nr_math.o $(OBJ)/ame_sync.o \
  $(OBJ)/launch.o $(OBJ)/adapter.o $(OBJ)/board_support.o $(OUT)/kernel.o

.PHONY: bin board-inputs-force
bin: $(BIN)
board-inputs-force:

$(OBJ):
	mkdir -p "$@"

# Preserve timestamps when commands are unchanged; overrides invalidate every
# runtime object and the image, including changes to the final audit policy.
$(BOARD_IDENTITY): board-inputs-force $(REPORTER) | $(OBJ)
	@$(PYTHON) -B "$(REPORTER)" inputs --inputs "$@" \
	  --value "clang=$(CLANG) $(BOARD_FLAGS)" --value 'dma=$(BOARD_DMA_FLAGS)' \
	  --value 'linker=$(LINKER) $(BOARD_LINK_SCRIPT)' \
	  --value 'objcopy=$(OBJCOPY)' --value 'objdump=$(OBJDUMP)' \
	  --value 'audit=$(PYTHON) $(BUDDY_TOOLS)/check_nr_elf.py $(FPGA_REDUCTION_AUDIT_FLAGS)'

$(filter $(OBJ)/%,$(IMAGE_OBJECTS)): $(BOARD_IDENTITY) $(BOARD_BUILD_INPUTS) $(CLANG_FILE)
-include $(filter $(OBJ)/%,$(IMAGE_OBJECTS:.o=.d))

define compile_board
	"$(CLANG)" $(BOARD_FLAGS) $(1) -MMD -MP -MF "$(@:.o=.d)" -c "$<" -o "$@"
endef

$(OBJ)/launch.o: $(CURDIR)/launch.c | $(OBJ)
	$(call compile_board,)
$(OBJ)/adapter.o: $(OUT)/adapter.c | $(OBJ)
	$(call compile_board,)
$(OBJ)/board_support.o: $(MODEL)/board_support.c | $(OBJ)
	$(call compile_board,)
$(OBJ)/nh_e6_boot.o: $(RUNTIME)/nh_e6/nh_e6_boot.S | $(OBJ)
	$(call compile_board,)
$(OBJ)/nh_e6_platform.o: $(RUNTIME)/nh_e6/nh_e6_platform.c | $(OBJ)
	$(call compile_board,-Dinit_uart=effnet_init_uart -DNH_E6_WAIT_ITERATIONS=0xffffffffu)
$(OBJ)/uart.o: $(RUNTIME)/uart/uart.c | $(OBJ)
	$(call compile_board,)
$(OBJ)/bosc_ame_dma.o: $(RUNTIME)/runtime/bosc_ame_dma.c | $(OBJ)
	$(call compile_board,$(BOARD_DMA_FLAGS))
$(OBJ)/nr_math.o: $(RUNTIME)/nr/nr_math.c | $(OBJ)
	$(call compile_board,)
$(OBJ)/ame_sync.o: $(RUNTIME)/nr/ame_sync.c | $(OBJ)
	$(call compile_board,)

$(ELF): $(IMAGE_OBJECTS) $(BOARD_IDENTITY) $(BOARD_BUILD_INPUTS) $(BOARD_LINK_SCRIPT) $(LINKER_FILE)
	"$(LINKER)" -m elf64lriscv --gc-sections -T "$(BOARD_LINK_SCRIPT)" -Map="$(OUT)/$(CASE).map" -o "$@" \
	  $(IMAGE_OBJECTS)
$(BIN): $(ELF) $(BOARD_IDENTITY) $(BUDDY_TOOLS)/check_nr_elf.py $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE) $(OBJCOPY_FILE) $(OBJDUMP_FILE)
	$(PYTHON) -B "$(BUDDY_TOOLS)/check_nr_elf.py" "$<" \
	  --objdump "$(OBJDUMP)" $(FPGA_REDUCTION_AUDIT_FLAGS) --output "$(OUT)/elf-audit.json"
	"$(OBJCOPY)" -O binary "$<" "$@"
	@echo built "$@"
