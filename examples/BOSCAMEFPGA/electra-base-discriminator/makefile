#
# Per-case Make rules for FPGA operator 'makefile' (model
# 'electra-base-discriminator').
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
.DEFAULT_GOAL := asm
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
KERNELS ?= $(ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
BUDDY_ROOT ?= $(abspath $(ROOT)/../../../buddy-mlir)
MODEL_CLANG ?= $(BUDDY_ROOT)/llvm/build/bin/clang
SCOPE := $(if $(filter undefined,$(origin CASES)),--all,)
CASES ?= $(notdir $(patsubst %/kernel.py,%,$(wildcard $(ROOT)/*/kernel.py)))
export PYTHONPATH := $(ROOT)/tools:$(PYTHONPATH)
CASE_ARGS := $(foreach case,$(CASES),--case $(case))
.PHONY: ttir boscame llvm asm bin hostcheck launchcheck modelcheck model-abicheck model-runtime-object model-bin model-host simcheck assets weights clean
ttir boscame llvm asm bin:
	+@status=0; for case in $(CASES); do $(MAKE) --no-print-directory -C "$(ROOT)/$$case" $@ PYTHON="$(PYTHON)" KERNELS="$(KERNELS)" || status=1; done; \
	  $(PYTHON) -B "$(ROOT)/tools/report.py" collect --output "$(KERNELS)" --stage "$(if $(filter bin,$@),asm,$@)" $(CASE_ARGS) $(SCOPE) || status=1; exit $$status
hostcheck:
	$(PYTHON) -B "$(ROOT)/tools/check_kernels.py" $(CASE_ARGS)
launchcheck:
	$(PYTHON) -B "$(ROOT)/tools/prepare_adapters.py" --output "$(KERNELS)"
	$(PYTHON) -B "$(ROOT)/tools/check_launch.py" $(CASE_ARGS) --build-dir "$(KERNELS)"
modelcheck:
	$(PYTHON) -B "$(ROOT)/tools/model_reference.py" $(if $(ASSETS),--assets "$(ASSETS)",)
model-abicheck:
	$(PYTHON) -B "$(ROOT)/tools/check_model_abi.py"
model-runtime-object:
	mkdir -p "$(ROOT)/build/model-runtime"
	$(MODEL_CLANG) --target=riscv64-unknown-elf -march=rv64imafdc_zicsr_zifencei -mabi=lp64d -mcmodel=medany -mno-relax -O2 -nostdlib -ffreestanding -fno-builtin -I"$(ROOT)" -c "$(ROOT)/model_runtime/model.c" -o "$(ROOT)/build/model-runtime/model.o"
simcheck:
	$(PYTHON) -B "$(ROOT)/tools/simcheck.py" $(CASE_ARGS) --output "$(KERNELS)"
assets:
	$(PYTHON) -B "$(ROOT)/tools/fetch_assets.py" $(if $(ASSETS),--output "$(ASSETS)",)
weights:
	$(PYTHON) -B "$(ROOT)/tools/prepare_weights.py" $(if $(ASSETS),--assets "$(ASSETS)",)
model-bin:
	$(PYTHON) -B "$(ROOT)/tools/prepare_model_board.py" $(if $(ASSETS),--assets "$(ASSETS)",)
	$(PYTHON) -B "$(ROOT)/tools/check_model_weights.py"
	$(PYTHON) -B "$(ROOT)/tools/build_model_bin.py" --kernels "$(KERNELS)"
model-host:
	$(PYTHON) -B "$(ROOT)/tools/build_model_host.py" --kernels "$(KERNELS)"
clean:
	rm -rf "$(ROOT)/build"
