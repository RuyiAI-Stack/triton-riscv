# Codex Operator Workflow for Triton-RISCV

This document defines a reusable workflow for guiding Codex to generate,
complete, compile, and test Triton operators for Triton-RISCV.

## Goal

The goal is not only to implement one operator. The goal is to create a
repeatable process that gives Codex enough context to work on Triton-RISCV
operators, then evaluate whether Codex follows the process successfully.

## Workflow Loop

1. Select an operator.
2. Collect operator context.
3. Generate a Codex task description.
4. Ask Codex to implement or complete the operator.
5. Build and test the result.
6. Record success, failure logs, and likely failure reasons.
7. Update this workflow based on what Codex missed.

The loop should be treated as an experiment cycle. A failed operator test does
not always mean the operator implementation is wrong. Codex must first identify
whether the failure is caused by environment setup, Python import, Triton
compilation, MLIR lowering, runtime execution, or numerical correctness.

## Operator Context

For each operator, collect the following information before asking Codex to
write code:

- Operator name
- Operator semantics
- PyTorch reference implementation
- Existing Triton or FlagGems reference implementation
- Expected input and output shapes
- Expected dtypes
- Files that Codex should inspect
- Files that Codex may modify
- Triton-RISCV limitations to avoid

## Task Generation Flow

Use the task generator to create an operator-specific Codex task file:

```sh
python codex/generate_operator_task.py <operator>
```

The generator reads `codex/templates/operator-task.md` and writes:

```text
tasks/operators/<operator>.md
```

Pass operator-specific context such as semantics, PyTorch reference,
implementation file, test file, and validation command through generator
arguments. See `codex/README.md` for an example.

After generation, Codex should complete the task file with the actual validation
attempts and any workflow feedback discovered during testing.

## Environment Prerequisite Check

Before building or testing an operator, Codex should check that the local
Triton-RISCV environment is available. If the environment is missing, record the
missing prerequisite instead of treating it as an operator failure.

Check the following items:

- `../triton` exists, or `TRITON_DIR` points to a valid Triton checkout
- `../buddy-mlir` exists, or `BUDDY_DIR` points to a valid Buddy MLIR checkout
- the Python environment exists, usually `.venv`
- `pytest` is available in the active environment
- `scripts/triton-riscv-env.sh` can be sourced without path errors
- the host architecture is supported by the chosen setup path; for example,
  `scripts/build.sh` currently supports `x86_64`/`amd64` and `riscv64`

If any prerequisite is missing, stop before running operator tests and record an
environment setup failure in the validation log.

## Suggested Files

Codex should usually inspect:

- `python/examples/`
- `python/examples/flaggems/`
- `docs/05-Operator Migration.md`
- `README.md`

Codex should usually modify or add:

- a Python example or operator implementation under `python/examples/`
- a pytest test case under `python/examples/`
- a task description under `tasks/operators/`

## Compilation and Test Commands

Use the repository environment helper before running build or test commands:

```sh
source scripts/triton-riscv-env.sh
```

Build or rebuild Triton-RISCV:

```sh
scripts/rebuild-triton-riscv.sh
```

Run the relevant operator test:

```sh
python -m pytest -q python/examples/flaggems/test_<operator>.py -s
```

Save each validation run to a timestamped log:

```sh
mkdir -p test-logs
log="test-logs/<operator>-$(date +%Y%m%d-%H%M%S).log"
python -m pytest -q python/examples/flaggems/test_<operator>.py -s 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "exit=$status" | tee -a "$log"
```

Use the saved log path in the validation record.

## Compilation Failure Triage

When a pytest case fails during compilation, Codex should classify the failure
before changing any implementation file.

Use this order:

1. Check whether the environment was loaded correctly.
2. Check whether the Python test imports the intended operator.
3. Check whether Triton JIT compilation started.
4. Check whether the failure happened in Triton-RISCV lowering or MLIR
   translation.
5. Check whether execution reached the PyTorch reference comparison.

If the log contains an error like:

```text
Dialect linalg not found for custom op linalg.generic
```

then record the failure as an MLIR lowering or translation failure. Do not treat
it as a numerical correctness failure. Codex should inspect the lowering
pipeline and whether `linalg.generic` remains before `mlir-translate
--mlir-to-llvmir`. In this case, the workflow improvement should be to add a
pre-translation IR check or to document the unsupported math/linalg lowering
path.

## Success Criteria

An operator attempt is successful when:

- the implementation follows the selected operator semantics
- the code compiles through the Triton-RISCV flow
- the pytest reference comparison passes
- the result is compared against a PyTorch reference implementation
- any limitations or unsupported cases are documented

A validation attempt can still be useful when it fails, as long as the workflow
records the exact failing stage, log file, and likely reason. In that case, the
workflow update is the deliverable, not an unproven code change.

## Failure Record

For each validation run, append a new attempt entry instead of replacing older
results. This keeps the workflow evolution visible across repeated tests.

For each failed attempt, record:

- operator name
- task file used
- command that failed
- short error log
- failure stage: generation, import, compilation, runtime, or correctness
- likely failure reason
- possible workflow improvement

For compilation failures, also record:

- whether the failing command was `triton-shared-opt`, `buddy-opt`,
  `mlir-translate`, `llc`, or pytest itself
- whether the generated MLIR still contains unsupported dialect operations such
  as `linalg.generic`
- whether the failure is shared by related operators, for example `gelu`
  failing together with standalone `erf`

## Initial Validation Operators

Start with simple operators before trying larger kernels:

- baseline: `vec_add`
- passing operator: `silu`
- mixed-result operator: `gelu`
- related failure-isolation operator: `erf`
- generated fused operator: `sigmoid_and_mul`
- generated fused operator: `relu_and_mul`
