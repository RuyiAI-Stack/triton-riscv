#
# Shared Make rules for BOSCAME FPGA model 'mobilenetv3_small_100.lamb_in1k'.
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
BOARD_PROFILE ?= nh-only
MODEL := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ifeq ($(BOARD_PROFILE),nh-e6)
include $(MODEL)/board_e6.mk
else ifeq ($(BOARD_PROFILE),nh-only)
CASE := $(notdir $(CURDIR))
REPO_ROOT := $(abspath $(MODEL)/../../..)
BUDDY_ROOT ?= $(REPO_ROOT)/buddy-mlir
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(BUDDY_ROOT)/llvm/build/bin)
CLANG ?= $(LLVM_BIN)/clang
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
LD := $(LINKER) -m elf64lriscv
OBJCOPY := $(LLVM_BIN)/llvm-objcopy
OBJDUMP := $(LLVM_BIN)/llvm-objdump
PYTHON ?= python3
KERNELS ?= $(MODEL)/build/kernels
override OUT := $(abspath $(KERNELS))/$(CASE)
OBJ := $(OUT)/obj-nh
ELF := $(OUT)/$(CASE).elf
BIN := $(OUT)/$(CASE).bin
NH := $(MODEL)/runtime/nh
CFLAGS := --target=riscv64-unknown-elf \
  -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections -ffp-contract=off \
  -fno-vectorize -fno-slp-vectorize -I$(MODEL) -I$(NH) \
  -DBOSC_NH_ONLY=1 -DUART_FREQ_HZ=5529600 -DVIT_WORKSPACE_BYTES=16777216 \
  -MMD -MP
RUNTIME_OBJECTS := $(addprefix $(OBJ)/,crt_uart.o bare_runtime.o uart.o uart_protocol.o)
IMAGE_OBJECTS := $(RUNTIME_OBJECTS) $(OBJ)/launch.o $(OBJ)/adapter.o \
  $(OBJ)/board_support.o $(OUT)/kernel.o
BUILD_FILES := $(MODEL)/board.mk $(MODEL)/support.h $(wildcard $(NH)/*.h)
-include $(filter $(OBJ)/%,$(IMAGE_OBJECTS:.o=.d))

.PHONY: bin FORCE
bin: $(BIN)
FORCE:
$(OBJ):
	mkdir -p "$@"
$(OUT)/.board-profile: FORCE | $(OBJ)
	@$(PYTHON) -B "$(MODEL)/tools/report.py" inputs --inputs "$@" --value "profile=nh-only"
$(OBJ)/.config: FORCE | $(OBJ)
	@printf '%s\n' '$(CLANG) $(CFLAGS)' '$(LD)' '$(OBJCOPY)' '$(OBJDUMP)' > "$@.tmp"
	@cmp -s "$@.tmp" "$@" || mv "$@.tmp" "$@"
	@rm -f "$@.tmp"
$(OBJ)/launch.o: $(CURDIR)/launch.c $(BUILD_FILES) $(OBJ)/.config
	$(CLANG) $(CFLAGS) -c "$<" -o "$@"
$(OBJ)/adapter.o: $(OUT)/adapter.c $(BUILD_FILES) $(OBJ)/.config
	$(CLANG) $(CFLAGS) -c "$<" -o "$@"
$(OBJ)/board_support.o: $(MODEL)/board_support.c $(BUILD_FILES) $(OBJ)/.config
	$(CLANG) $(CFLAGS) -c "$<" -o "$@"
$(OBJ)/crt_uart.o: $(NH)/crt_uart.S $(BUILD_FILES) $(OBJ)/.config
	$(CLANG) $(CFLAGS) -c "$<" -o "$@"
$(OBJ)/%.o: $(NH)/%.c $(BUILD_FILES) $(OBJ)/.config
	$(CLANG) $(CFLAGS) -c "$<" -o "$@"
$(ELF): $(IMAGE_OBJECTS) $(MODEL)/link_nh.ld $(OUT)/.board-profile $(OBJ)/.config
	@rm -f "$(BIN)"
	$(LD) --gc-sections -T "$(MODEL)/link_nh.ld" -Map="$(OUT)/$(CASE).map" \
	  -o "$@" $(IMAGE_OBJECTS)
$(BIN): $(ELF) $(NH)/audit.py
	$(PYTHON) -B "$(NH)/audit.py" "$<" --objdump "$(OBJDUMP)" --output "$(OUT)/elf-audit.json"
	$(OBJCOPY) -O binary "$<" "$@"
	@echo built $@
else
$(error Unsupported BOARD_PROFILE=$(BOARD_PROFILE); choose nh-only or nh-e6)
endif
