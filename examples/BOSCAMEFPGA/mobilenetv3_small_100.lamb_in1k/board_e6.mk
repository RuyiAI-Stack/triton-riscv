# Link one case's kernel.o into an NH/E6 image. Does not compile the kernel.
# Usage, from a case directory: make -f ../board.mk bin
CASE := $(notdir $(CURDIR))
MODEL := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(MODEL)/../../..)
BUDDY_ROOT ?= $(abspath $(REPO_ROOT)/buddy-mlir)
REPO := $(BUDDY_ROOT)
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(REPO)/llvm/build/bin)
LLVM := $(LLVM_BIN)
RUNTIME := $(REPO)/runtime/boscame
PYTHON ?= python3
KERNELS ?= $(MODEL)/build/kernels
override OUT := $(abspath $(KERNELS))/$(CASE)
OBJ := $(OUT)/obj
ELF := $(OUT)/$(CASE).elf
BIN := $(OUT)/$(CASE).bin

CLANG ?= $(LLVM)/clang
CC := $(CLANG) --target=riscv64-unknown-elf
LINKER ?= $(firstword $(wildcard $(LLVM)/ld.lld) ld.lld)
LD := $(LINKER) -m elf64lriscv
OBJCOPY := $(LLVM)/llvm-objcopy
OBJDUMP := $(LLVM)/llvm-objdump
CFLAGS := -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany \
  -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -fno-pie \
  -ffunction-sections -fdata-sections -ffp-contract=off \
  -fno-vectorize -fno-slp-vectorize \
  -I$(MODEL) -I$(RUNTIME)/runtime -I$(RUNTIME)/uart \
  -I$(RUNTIME)/nh_e6 -I$(RUNTIME)/nr \
  '-DBOSC_AME_DMA_ASSERT(x)=' -DUART_FREQ_HZ=5529600 \
  -DVIT_WORKSPACE_BYTES=16777216 -MMD -MP

IMAGE_OBJECTS := $(OBJ)/nh_e6_boot.o $(OBJ)/nh_e6_platform.o $(OBJ)/uart.o \
  $(OBJ)/bosc_ame_dma.o $(OBJ)/nr_math.o $(OBJ)/ame_sync.o \
  $(OBJ)/launch.o $(OBJ)/adapter.o $(OBJ)/board_support.o $(OUT)/kernel.o

.DELETE_ON_ERROR:
BUILD_FILES := $(MODEL)/board.mk $(MODEL)/board_e6.mk $(MODEL)/support.h
-include $(IMAGE_OBJECTS:.o=.d)

.PHONY: bin FORCE
bin: $(BIN)
FORCE:

$(OBJ):
	mkdir -p $@

$(OUT)/.board-profile: FORCE | $(OBJ)
	@$(PYTHON) -B "$(MODEL)/tools/report.py" inputs --inputs "$@" --value "profile=nh-e6"
$(OBJ)/.config: FORCE | $(OBJ)
	@$(PYTHON) -B "$(MODEL)/tools/report.py" inputs --inputs "$@" \
	  --value "cc=$(CC) $(CFLAGS)" --value "ld=$(LD)" --value "objcopy=$(OBJCOPY)"
$(filter $(OBJ)/%,$(IMAGE_OBJECTS)): $(BUILD_FILES) $(OBJ)/.config

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

$(ELF): $(IMAGE_OBJECTS) $(MODEL)/link.ld $(OUT)/.board-profile $(OBJ)/.config
	@rm -f "$(BIN)"
	$(LD) --gc-sections -T $(MODEL)/link.ld -Map=$(OUT)/$(CASE).map -o $@ \
	  $(IMAGE_OBJECTS)
$(BIN): $(ELF) $(REPO)/examples/FPGA-BOSCAME/tools/check_nr_elf.py
	$(PYTHON) -B $(REPO)/examples/FPGA-BOSCAME/tools/check_nr_elf.py $< \
	  --objdump $(OBJDUMP) --output $(OUT)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
	@echo built $@
