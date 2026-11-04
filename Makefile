SHELL := /bin/bash

ARCH   ?= $(shell uname -m | sed 's/^amd64$$/x86_64/')
PYTEST ?= python/examples
IGNORE := --ignore=python/examples/test_core.py --ignore=python/examples/test_annotations.py --ignore=python/examples/flaggems

PY_VER  ?= $(shell .venv/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
LIT_DIR := triton/build/cmake.linux-$(ARCH)-cpython-$(PY_VER)

.PHONY: all setup build test lit clean

all: setup build

setup:
	scripts/setup.sh

build:
	scripts/build.sh

ifeq ($(ARCH),riscv64)
test: lit
	bash -c 'source scripts/env.sh && pytest $(PYTEST) $(IGNORE) -q && pytest python/examples/flaggems -q'
else
test: lit
	bash -c 'source scripts/env.sh && pytest $(PYTEST) $(IGNORE) -q'
endif

lit:
	bash -c 'source scripts/env.sh && cmake --build "$(LIT_DIR)" --target check-triton-shared-lit-tests'

clean:
	rm -rf triton/build
