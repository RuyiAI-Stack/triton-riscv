# Build real Triton operators; compiler recipes live in common.mk.
.DEFAULT_GOAL := all
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON ?= python3
KERNELS ?= $(ROOT)/build/kernels
KERNELS := $(abspath $(KERNELS))
BUDDY_ROOT ?= $(abspath $(ROOT)/../../../buddy-mlir)
export BUDDY_ROOT
export PYTHONPATH := $(ROOT)/tools:$(PYTHONPATH)
ALL_CASES := $(notdir $(patsubst %/kernel.py,%,$(wildcard $(ROOT)/*/kernel.py)))
# An explicit CASES selection writes a subset report, preserving suite evidence.
SCOPE := $(if $(filter undefined,$(origin CASES)),--all,)
CASES ?= $(ALL_CASES)
CASE_ARGS = $(foreach case,$(sort $(CASES)),--case $(case))
CASE_TARGETS := $(addprefix build-,$(sort $(CASES)))
REPORTER := $(ROOT)/tools/report.py
FRONTEND := $(abspath $(ROOT)/../tools/contract_frontend.py)
FRONTEND_ID := $(KERNELS)/.frontend-inputs.json
STAGE ?= asm

.PHONY: all ttir boscame llvm asm _suite $(CASE_TARGETS) FORCE simcheck clean
all: asm
ttir boscame llvm asm:
	+@code=0; $(MAKE) --no-print-directory -k _suite STAGE=$@ || code=$$?; \
	  $(PYTHON) -B "$(REPORTER)" collect --output "$(KERNELS)" --stage "$@" $(CASE_ARGS) $(SCOPE) $(if $(REPORT),--report "$(REPORT)") || code=1; \
	  exit $$code

# -k attempts every selected case, including when another case fails. Recursive
# Make shares the parent's jobserver, so make -jN also builds cases in parallel.
_suite: $(CASE_TARGETS)
$(CASE_TARGETS): build-%: $(FRONTEND_ID)
	+@rm -f "$(KERNELS)/$*/build-result.json"; \
	  $(MAKE) --no-print-directory -C "$(ROOT)/$*" $(STAGE) KERNELS="$(KERNELS)" PYTHON="$(PYTHON)" FRONTEND_CHECKED=1 REPORT= || { \
	    if ! test -f "$(KERNELS)/$*/build-result.json"; then \
	      $(PYTHON) -B "$(REPORTER)" failure --case "$*" --output "$(KERNELS)" --stage "$(STAGE)" || :; \
	    fi; exit 1; }
FORCE:
$(KERNELS):
	mkdir -p "$@"
$(FRONTEND_ID): FORCE | $(KERNELS)
	@$(PYTHON) -B "$(FRONTEND)" --identity "$@" > "$(KERNELS)/frontend-identity.log" 2>&1 || { \
	  cat "$(KERNELS)/frontend-identity.log" >&2; \
	  for name in $(CASES); do \
	    $(PYTHON) -B "$(REPORTER)" failure --case "$$name" --output "$(KERNELS)" --stage ttir --log "$(KERNELS)/frontend-identity.log" || :; \
	  done; exit 1; }

simcheck:
	$(PYTHON) -B "$(ROOT)/tools/simcheck.py" $(if $(SCOPE),--all,$(CASE_ARGS)) --output "$(KERNELS)" $(if $(COLLECT),--collect,)
clean:
	rm -rf "$(ROOT)/build"
