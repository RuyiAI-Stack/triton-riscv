#
# Shared Make rules for BOSCAME FPGA model 'xlnet-base-cased'.
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
BOARD_GENERATOR := $(VIT_ROOT)/tools/board_config.py
BOARD_SUPPORT := $(VIT_ROOT)/tools/board_support.c
BOARD_HEADER := $(VIT_ROOT)/tools/board_support.h
BOARD_DEP_CASES := $(shell $(PYTHON) -B $(BOARD_GENERATOR) --case $(CASE) --dependencies)
BOARD_LINK_SCRIPT ?= $(VIT_ROOT)/board.ld
SCHEDULER_OUTPUT_TOOLS := $(REPO_ROOT)/examples/BOSCAMEFPGA/tools
BOARD_BUILD_INPUTS := $(VIT_ROOT)/board.mk $(BOARD_LINK_SCRIPT) $(CURDIR)/makefile \
  $(VIT_ROOT)/tools/attention_reference.h $(VIT_ROOT)/tools/launch_checks.h \
  $(SCHEDULER_OUTPUT_TOOLS)/scheduler_outputs.py $(VIT_ROOT)/tools/output_spec.py \
  $(SCHEDULER_OUTPUT_TOOLS)/scheduler_output_runtime.h $(SCHEDULER_OUTPUT_TOOLS)/scheduler_output_runtime.inc
# This makefile links the delivered single-executor NH/E6 platform explicitly.
# Keep the generic runtime default atomic; only dma.o selects the E6 guard.
BOARD_PRIVATE_DDR_NR_SINGLE_E6 ?= 1
ifneq ($(words $(BOARD_PRIVATE_DDR_NR_SINGLE_E6)),1)
$(error BOARD_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1)
endif
ifeq ($(filter 0 1,$(BOARD_PRIVATE_DDR_NR_SINGLE_E6)),)
$(error BOARD_PRIVATE_DDR_NR_SINGLE_E6 must be 0 or 1)
endif
BOARD_DMA_FLAGS := -DBOSC_AME_PRIVATE_DDR_NR_SINGLE_E6=$(BOARD_PRIVATE_DDR_NR_SINGLE_E6)
BOARD_LAUNCH_SOURCE := $(BOARD)/board_launch.c
BOARD_FLAGS := --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -fno-stack-protector -fno-vectorize -fno-slp-vectorize \
  -ffp-contract=off -ffunction-sections -fdata-sections \
  '-DBOSC_AME_DMA_ASSERT(x)=' \
  -I$(SCHEDULER_OUTPUT_TOOLS) -I$(VIT_ROOT) -I$(VIT_ROOT)/tools -I$(BOARD) -I$(PLATFORM) -I$(DMA_RUNTIME) -I$(UART) -I$(NR)
BOARD_OBJECTS := $(addprefix $(BOARD)/,launch.o adapter.o wrapper.o support.o boot.o platform.o uart.o dma.o math.o) $(OUT)/kernel.o \
  $(foreach dep,$(BOARD_DEP_CASES),$(BOARD)/$(dep)-adapter.o $(KERNELS)/$(dep)/kernel.o) $(BOARD_OBJECTS_EXTRA)
BOARD_IDENTITY := $(BOARD)/.board-inputs.json
$(BOARD_IDENTITY): FORCE $(REPORTER) | $(BOARD)
	@$(PYTHON) -B $(REPORTER) inputs --inputs $@ --value "clang=$(CLANG) $(BOARD_FLAGS)" --value 'linker=$(LINKER) $(BOARD_LINK_SCRIPT)' --value 'dma=$(BOARD_DMA_FLAGS)' --value 'output-config=$(BOARD_CONFIG_FLAGS)'
$(filter $(BOARD)/%,$(BOARD_OBJECTS)): $(BOARD_IDENTITY)

$(BOARD):
	mkdir -p "$@"
$(BOARD)/board_case.h $(BOARD)/board_wrapper.c $(BOARD)/board_config.json $(BOARD)/board_launch.c &: $(CURDIR)/kernel.py $(CURDIR)/launch.c $(VIT_ROOT)/cases.py $(BOARD_GENERATOR) $(BOARD_IDENTITY) $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(PYTHON) -B $(BOARD_GENERATOR) --case $(CASE) --output $(BOARD) $(BOARD_CONFIG_FLAGS)
$(BOARD)/launch.o: $(BOARD_LAUNCH_SOURCE) ../support.h $(BOARD_HEADER) $(BOARD)/board_case.h $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -include $(BOARD_HEADER) -c $< -o $@
$(BOARD)/adapter.c: $(OUT)/adapter.c $(BOARD_GENERATOR) $(CURDIR)/kernel.py $(BOARD_IDENTITY) | $(BOARD)
	$(PYTHON) -B $(BOARD_GENERATOR) --case $(CASE) --output $(BOARD) --adapter-source $< $(BOARD_CONFIG_FLAGS)
$(BOARD)/adapter.o: $(BOARD)/adapter.c ../support.h $(BOARD_HEADER) $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -include $(BOARD_HEADER) -D_mlir_ciface_kernel_$(CASE)=vit_board_impl_$(CASE) -c $< -o $@
$(BOARD)/wrapper.o: $(BOARD)/board_wrapper.c $(BOARD_HEADER) $(BOARD_BUILD_INPUTS)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/support.o: $(BOARD_SUPPORT) $(BOARD_HEADER) ../support.h $(BOARD)/board_case.h $(BOARD_BUILD_INPUTS)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/boot.o: $(PLATFORM)/nh_e6_boot.S $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/platform.o: $(PLATFORM)/nh_e6_platform.c $(PLATFORM)/nh_e6_platform.h $(BOARD_HEADER) $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -include $(BOARD_HEADER) -Dinit_uart=board_init_uart \
	  -DNH_E6_WAIT_ITERATIONS=0xffffffffu -DNH_E6_MMIO_READ32=vit_board_mmio_read32 -c $< -o $@
$(BOARD)/uart.o: $(UART)/uart.c $(UART)/uart.h $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@
$(BOARD)/dma.o: $(DMA_RUNTIME)/bosc_ame_dma.c $(DMA_RUNTIME)/bosc_ame_dma.h $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) $(BOARD_DMA_FLAGS) -c $< -o $@
$(BOARD)/math.o: $(NR)/nr_math.c $(NR)/nr_runtime.h $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -c $< -o $@

# Always enter dependency builds so their command manifest sees the inherited
# reduction policy, including a dependency rebuilt separately with an opt-out.
define board_dependency
$(KERNELS)/$(1)/kernel.o $(KERNELS)/$(1)/adapter.c &: $(VIT_ROOT)/$(1)/kernel.py $(BUILD_FILES) $(EXPORTER) $(VIT_ROOT)/cases.py FORCE
	+$$(MAKE) -C $(VIT_ROOT)/$(1) asm KERNELS=$(KERNELS)
$(BOARD)/$(1)-adapter.o: $(KERNELS)/$(1)/adapter.c ../support.h $(BOARD_BUILD_INPUTS) | $(BOARD)
	$(CLANG) $(BOARD_FLAGS) -D_mlir_ciface_kernel_$(1)=vit_board_impl_$(1) -c $$< -o $$@
endef
$(foreach dep,$(BOARD_DEP_CASES),$(eval $(call board_dependency,$(dep))))

$(BOARD_ELF): $(BOARD_OBJECTS) $(BOARD_LINK_SCRIPT) $(BOARD_BUILD_INPUTS)
	$(LINKER) -m elf64lriscv --gc-sections -T $(BOARD_LINK_SCRIPT) -Map=$(BOARD)/$(CASE).map -o $@ $(BOARD_OBJECTS)
$(BOARD_BIN): $(BOARD_ELF) $(BUDDY_TOOLS)/check_nr_elf.py $(OUT)/.ordered-reduction-policy $(FPGA_REDUCTION_POLICY_MAKEFILE)
	$(PYTHON) $(BUDDY_TOOLS)/check_nr_elf.py $< --objdump $(OBJDUMP) $(FPGA_REDUCTION_AUDIT_FLAGS) --output $(BOARD)/elf-audit.json
	$(OBJCOPY) -O binary $< $@
.PHONY: bin
bin: $(BOARD_BIN)
