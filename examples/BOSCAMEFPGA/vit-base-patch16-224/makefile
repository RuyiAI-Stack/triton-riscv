# Build each real Triton operator through contract -> plan -> verify -> emit.
.DEFAULT_GOAL := all
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
ASSETS ?= $(ROOT)/assets
KERNELS ?= $(ROOT)/build/kernels
IMPORT ?= $(ROOT)/build/import
MODE ?= w8a8
REPLACEMENT ?= $(ROOT)/build/replacement-$(MODE)
.PHONY: all ttir boscame llvm asm simcheck w8a8-chaincheck model-simcheck launchcheck assets import replace lower clean
all: asm
ttir boscame llvm asm:
	$(PYTHON) -B $(ROOT)/tools/build.py --all --stage $@ --output $(KERNELS)
simcheck:
	$(PYTHON) -B $(ROOT)/tools/simcheck.py --all --output $(KERNELS) $(if $(COLLECT),--collect,)
w8a8-chaincheck:
	$(PYTHON) -B $(ROOT)/tools/build.py --stage boscame --output $(KERNELS) --case quantize_1x768 --case quantize_1000x768 --case matmul_i8_overwrite_1x1000x768 --case dequantize_1x1000 --case bias_add_1x1000 --case add_1x1000
	$(PYTHON) -B $(ROOT)/tools/check_w8a8_chain.py --kernels $(KERNELS)
model-simcheck:
	$(PYTHON) -B $(ROOT)/tools/model_simcheck.py --kernels $(KERNELS) --mode $(MODE)
launchcheck:
	$(PYTHON) -B $(ROOT)/tools/check_launch.py --all --build-dir $(KERNELS)
assets:
	$(PYTHON) -B $(ROOT)/tools/fetch_assets.py --output $(ASSETS)
import:
	$(PYTHON) -B $(ROOT)/tools/import_model.py --assets $(ASSETS) --checkpoint $(ASSETS)/model.safetensors --output $(IMPORT) --save-mlir
replace:
	$(PYTHON) -B $(ROOT)/tools/replace_triton.py --import-dir $(IMPORT) --triton-build $(KERNELS) --output $(REPLACEMENT) --require-complete $(if $(filter w8a8,$(MODE)),--w8a8,--fp32-reference)
lower:
	$(PYTHON) -B $(ROOT)/tools/lower_model.py --replacement $(REPLACEMENT) --output $(ROOT)/build/lowered
clean:
	rm -rf $(ROOT)/build
