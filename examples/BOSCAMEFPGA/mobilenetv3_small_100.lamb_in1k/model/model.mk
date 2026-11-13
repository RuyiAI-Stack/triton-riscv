# Invoke from any directory: make -f /path/to/model/model.mk bin
.DEFAULT_GOAL := bin
.DELETE_ON_ERROR:
MODEL_SOURCE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
MODEL_ROOT := $(abspath $(MODEL_SOURCE)/..)
REPO_ROOT := $(abspath $(MODEL_ROOT)/../../..)
PYTHON ?= python3
BUDDY_ROOT ?= $(REPO_ROOT)/buddy-mlir
LLVM_BIN ?= $(or $(LLVM_BINARY_DIR),$(BUDDY_ROOT)/llvm/build/bin)
CLANG ?= $(LLVM_BIN)/clang
LLC ?= $(LLVM_BIN)/llc
LINKER ?= $(firstword $(wildcard $(LLVM_BIN)/ld.lld) ld.lld)
LD := $(LINKER)
OBJCOPY := $(LLVM_BIN)/llvm-objcopy
OBJDUMP := $(LLVM_BIN)/llvm-objdump
KERNELS ?= $(MODEL_ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
OUT ?= $(MODEL_ROOT)/build/model
OUT := $(abspath $(OUT))
OBJ := $(OUT)/obj
RUNTIME := $(MODEL_ROOT)/runtime/nh
# A bare `bin` invocation preserves an existing export's selected image and
# checkpoint. Explicit IMAGE= switches back to the deterministic fixture.
CHECKPOINT ?= $(shell $(PYTHON) -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()).get("checkpoint", "") if p.exists() else ""; print(s if pathlib.Path(s).is_file() else "")' "$(OUT)/model-manifest.json")
IMAGE ?= $(shell $(PYTHON) -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); print(json.loads(p.read_text()).get("input", {}).get("path", "") if p.exists() else "")' "$(OUT)/model-manifest.json")
JOBS ?= 8

CFLAGS := --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei \
  -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding \
  -fno-builtin -fno-pie -ffunction-sections -fdata-sections -ffp-contract=off \
  -fno-vectorize -fno-slp-vectorize -I$(MODEL_ROOT) -I$(MODEL_SOURCE) -I$(RUNTIME) \
  -DUART_FREQ_HZ=5529600 -MMD -MP

.PHONY: FORCE
FORCE:
$(OUT)/.export-inputs.json: FORCE
	@$(PYTHON) -B $(MODEL_ROOT)/tools/report.py inputs --inputs "$@" \
	  --value 'checkpoint=$(CHECKPOINT)' --value 'image=$(IMAGE)'
$(OUT)/.compiler-inputs.json: FORCE
	@$(PYTHON) -B $(MODEL_ROOT)/tools/report.py inputs --inputs "$@" \
	  --value 'clang=$(CLANG) $(CFLAGS)' --value 'ld=$(LD)' --value 'objcopy=$(OBJCOPY)' --value 'objdump=$(OBJDUMP)'

GENERATED := $(addprefix $(OUT)/,runner.c model_blob.S model-cases.mk model-manifest.json model-data.bin golden-logits.f32)
$(GENERATED) &: $(MODEL_ROOT)/tools/export_model.py $(MODEL_ROOT)/tools/export.py \
  $(MODEL_ROOT)/cases.py $(wildcard $(MODEL_ROOT)/*/kernel.py) $(CHECKPOINT) $(IMAGE) \
  $(OUT)/.export-inputs.json
	$(PYTHON) -B $(MODEL_ROOT)/tools/export_model.py --output $(OUT) \
	  $(if $(CHECKPOINT),--checkpoint "$(CHECKPOINT)") $(if $(IMAGE),--image "$(IMAGE)")

-include $(OUT)/model-cases.mk

KERNEL_OBJECTS := $(foreach case,$(MODEL_CASES),$(KERNELS)/$(case)/kernel.o)
ADAPTER_OBJECTS := $(foreach case,$(MODEL_CASES),$(OBJ)/adapter_$(case).o)
RUNTIME_OBJECTS := $(addprefix $(OBJ)/,crt_uart.o bare_runtime.o uart.o uart_protocol.o)
OBJECTS := $(OBJ)/runner.o $(OBJ)/model_blob.o $(OBJ)/model_support.o \
  $(OBJ)/board_support.o $(RUNTIME_OBJECTS) $(ADAPTER_OBJECTS) $(KERNEL_OBJECTS)
OWN_OBJECTS := $(filter-out $(KERNEL_OBJECTS),$(OBJECTS))
$(OWN_OBJECTS): $(MODEL_SOURCE)/model.mk $(OUT)/.compiler-inputs.json
-include $(OWN_OBJECTS:.o=.d)

.PHONY: export host bin kernels
export: $(GENERATED)
host: export $(KERNEL_OBJECTS)
	$(PYTHON) -B $(MODEL_ROOT)/tools/check_model_host.py --model $(OUT) \
	  --kernels $(KERNELS) --clang $(CLANG) --jobs $(JOBS)
# Re-enter per-case Makefiles so source changes and effective compiler policy
# invalidate existing kernel objects, including artifacts built elsewhere.
define model_kernel
$(KERNELS)/$(1)/kernel.o $(KERNELS)/$(1)/adapter.c &: FORCE
	+$$(MAKE) --no-print-directory -C $(MODEL_ROOT)/$(1) asm \
	  KERNELS=$(KERNELS) OUT=$(KERNELS)/$(1) PYTHON=$(PYTHON) BUDDY_ROOT=$(BUDDY_ROOT) LLVM_BIN=$(LLVM_BIN)
endef
$(foreach case,$(MODEL_CASES),$(eval $(call model_kernel,$(case))))
kernels: $(KERNEL_OBJECTS)
bin: $(OUT)/mobilenet.bin

$(OBJ):
	mkdir -p $@
$(OBJ)/runner.o: $(OUT)/runner.c $(MODEL_SOURCE)/model_support.h | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/model_blob.o: $(OUT)/model_blob.S $(OUT)/model-data.bin | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/model_support.o: $(MODEL_SOURCE)/model_support.c $(MODEL_SOURCE)/model_support.h | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/board_support.o: $(MODEL_SOURCE)/board_support.c | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/adapter_%.o: $(KERNELS)/%/adapter.c $(MODEL_ROOT)/support.h | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/crt_uart.o: $(RUNTIME)/crt_uart.S | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@
$(OBJ)/%.o: $(RUNTIME)/%.c | $(OBJ)
	$(CLANG) $(CFLAGS) -c $< -o $@

$(OUT)/mobilenet.elf: $(OBJECTS) $(MODEL_SOURCE)/link.ld
	@rm -f "$(OUT)/mobilenet.bin"
	$(LD) -m elf64lriscv --gc-sections -T $(MODEL_SOURCE)/link.ld \
	  -Map=$(OUT)/mobilenet.map -o $@ $(OBJECTS)
$(OUT)/mobilenet.bin: $(OUT)/mobilenet.elf $(RUNTIME)/audit.py $(BUDDY_ROOT)/examples/FPGA-BOSCAME/tools/check_nr_elf.py
	$(PYTHON) -B $(BUDDY_ROOT)/examples/FPGA-BOSCAME/tools/check_nr_elf.py $< \
	  --objdump $(OBJDUMP) --output $(OUT)/elf-audit.json
	$(PYTHON) -B $(RUNTIME)/audit.py $< \
	  --objdump $(OBJDUMP) --output $(OUT)/nh-audit.json
	$(OBJCOPY) -O binary $< $@
