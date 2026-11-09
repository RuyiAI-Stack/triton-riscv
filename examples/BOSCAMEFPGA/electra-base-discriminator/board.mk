#
# Shared Make rules for BOSCAME FPGA model 'electra-base-discriminator'.
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

CASE := $(notdir $(CURDIR))
MODEL := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MODEL)/../../..)
BUDDY_ROOT ?= $(abspath $(REPO_ROOT)/buddy-mlir)
REPO := $(BUDDY_ROOT)
LLVM := $(REPO)/llvm/build/bin
RUNTIME := $(REPO)/runtime/boscame
PYTHON ?= python3
OUT := $(MODEL)/build/kernels/$(CASE)
OBJ := $(OUT)/obj
ELF := $(OUT)/$(CASE).elf
BIN := $(OUT)/$(CASE).bin

CC := $(LLVM)/clang --target=riscv64-unknown-elf
LD := $(LLVM)/ld.lld -m elf64lriscv
OBJCOPY := $(LLVM)/llvm-objcopy
OBJDUMP := $(LLVM)/llvm-objdump
CFLAGS := -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections -ffp-contract=off \
  -fno-vectorize -fno-slp-vectorize \
  -I$(MODEL) -I$(RUNTIME)/runtime -I$(RUNTIME)/uart \
  -I$(RUNTIME)/nh_e6 -I$(RUNTIME)/nr \
  '-DBOSC_AME_DMA_ASSERT(x)=' -DUART_FREQ_HZ=5529600 \
  -DVIT_WORKSPACE_BYTES=16777216

IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o $(OBJ)/uart.o \
  $(OBJ)/bosc_ame_dma.o $(OBJ)/nr_math.o $(OBJ)/ame_sync.o \
  $(OBJ)/launch.o $(OBJ)/adapter.o $(OBJ)/board_support.o $(OUT)/kernel.o

.PHONY: bin
bin: $(BIN)

$(OBJ):
	mkdir -p $@

$(OBJ)/launch.o: $(CURDIR)/launch.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/adapter.o: $(OUT)/adapter.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/board_support.o: $(MODEL)/board_support.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/nh_e6_boot.o: $(RUNTIME)/nh_e6/nh_e6_boot.S | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/nh_e6_platform.o: $(RUNTIME)/nh_e6/nh_e6_platform.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/uart.o: $(RUNTIME)/uart/uart.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/bosc_ame_dma.o: $(RUNTIME)/runtime/bosc_ame_dma.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/nr_math.o: $(RUNTIME)/nr/nr_math.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJ)/ame_sync.o: $(RUNTIME)/nr/ame_sync.c | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(ELF): $(IMAGE_OBJECTS) $(MODEL)/link.ld
	$(LD) --gc-sections -T $(MODEL)/link.ld -Map=$(OUT)/$(CASE).map -o $@ \
	  $(IMAGE_OBJECTS)
$(BIN): $(ELF)
	$(PYTHON) -B $(REPO)/examples/FPGA-BOSCAME/tools/check_nr_elf.py $< \
	  --objdump $(OBJDUMP) --output $(OUT)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	@echo built $@
