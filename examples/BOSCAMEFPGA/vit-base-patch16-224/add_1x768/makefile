#
# Per-case Make rules for FPGA operator 'add_1x768' (model
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
include ../common.mk
.DEFAULT_GOAL := bin

# Board image: same NH/E6 boot, PMP window and DMA runtime as the standalone
# matmul_i8_128x1024x128 example. Keep the generic C compiler scalar-only.
BOARD := $(OUT)/board
PLATFORM := $(BUDDY_ROOT)/runtime/boscame/nh_e6
DMA_RUNTIME := $(BUDDY_ROOT)/runtime/boscame/runtime
UART := $(BUDDY_ROOT)/runtime/boscame/uart
NR := $(BUDDY_ROOT)/runtime/boscame/nr
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
OBJCOPY ?= $(LLVM_BIN)/llvm-objcopy
OBJDUMP ?= $(LLVM_BIN)/llvm-objdump
BOARD_ELF := $(BOARD)/$(CASE).elf
BOARD_BIN := $(BOARD)/$(CASE).bin
BOARD_FLAGS := --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -fno-stack-protector -fno-vectorize -fno-slp-vectorize \
  -ffp-contract=off -ffunction-sections -fdata-sections \
  -DVIT_WORKSPACE_BYTES=12288 '-DBOSC_AME_DMA_ASSERT(x)=' \
  -I$(VIT_ROOT) -I$(PLATFORM) -I$(DMA_RUNTIME) -I$(UART) -I$(NR)
BOARD_OBJECTS := $(addprefix $(BOARD)/,launch.o adapter.o board.o boot.o platform.o uart.o dma.o) $(OUT)/kernel.o

$(BOARD):
	mkdir -p "$@"
$(BOARD)/launch.o: launch.c ../support.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -include $(NR)/nr_runtime.h -c $< -o $@
$(BOARD)/adapter.o: $(OUT)/adapter.c ../support.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -D_mlir_ciface_kernel_add_1x768=vit_add_implementation -c $< -o $@
$(BOARD)/board.o: board.c ../support.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/boot.o: $(PLATFORM)/nh_e6_boot.S makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/platform.o: $(PLATFORM)/nh_e6_platform.c $(PLATFORM)/nh_e6_platform.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -Dinit_uart=board_init_uart -DNH_E6_WAIT_ITERATIONS=0xffffffffu -c $< -o $@
$(BOARD)/uart.o: $(UART)/uart.c $(UART)/uart.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/dma.o: $(DMA_RUNTIME)/bosc_ame_dma.c $(DMA_RUNTIME)/bosc_ame_dma.h makefile | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD_ELF): $(BOARD_OBJECTS) link.ld makefile
	$(LINKER) -m elf64lriscv --gc-sections -T link.ld -Map=$(BOARD)/$(CASE).map -o $@ $(BOARD_OBJECTS)
$(BOARD_BIN): $(BOARD_ELF) $(BUDDY_TOOLS)/check_nr_elf.py $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE)
	$(PYTHON) $(BUDDY_TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) $(FPGA_REDUCTION_AUDIT_FLAGS) --output $(BOARD)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
.PHONY: bin
bin: $(BOARD_BIN)
