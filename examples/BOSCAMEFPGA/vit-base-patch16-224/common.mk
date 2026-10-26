# Every case owns its Triton kernel and independent C launch; stages are shared.
.DEFAULT_GOAL := asm
CASE := $(notdir $(CURDIR))
VIT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
.PHONY: all ttir boscame llvm asm simcheck clean
all: asm
ttir boscame llvm asm:
	$(PYTHON) -B $(VIT_ROOT)/tools/build.py --case $(CASE) --stage $@
simcheck: boscame
	$(PYTHON) -B $(VIT_ROOT)/tools/simcheck.py --case $(CASE)
clean:
	rm -rf $(VIT_ROOT)/build/kernels/$(CASE)
