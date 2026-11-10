SHELL := /bin/bash

ARCH   ?= $(shell uname -m | sed 's/^amd64$$/x86_64/')
PYTHON ?= python3
PYTEST ?= python/examples
PYTEST_WORKERS ?= 16
IGNORE := --ignore=python/examples/test_core.py --ignore=python/examples/test_annotations.py --ignore=python/examples/flaggems

PY_VER  ?= $(shell $(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
LIT_DIR := triton/build/cmake.linux-$(ARCH)-cpython-$(PY_VER)

.PHONY: all setup build test test-lit test-tutorials clean

all: setup build

setup:
	scripts/setup.sh

build:
	scripts/build.sh

ifeq ($(ARCH),riscv64)
test: test-lit test-tutorials
	bash -c "source scripts/env.sh && $(PYTHON) -m pytest $(PYTEST) $(IGNORE) -q -n $(PYTEST_WORKERS) && $(PYTHON) -m pytest python/examples/flaggems -q -n $(PYTEST_WORKERS)"
else
test: test-lit test-tutorials
	bash -c "source scripts/env.sh && $(PYTHON) -m pytest $(PYTEST) $(IGNORE) -q -n $(PYTEST_WORKERS)"
endif

test-lit:
	bash -c 'source scripts/env.sh && cmake --build "$(LIT_DIR)" --target check-triton-shared-lit-tests'

clean:
	rm -rf triton/build

test-tutorials:
	bash -c 'source scripts/env.sh && $(PYTHON) -m pytest --nbmake python/tutorials -q'
