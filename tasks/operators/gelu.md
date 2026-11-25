# Operator Task: gelu

## Objective

Use `gelu` as a validation operator to evaluate whether the Codex workflow can
handle a multi-path elementwise activation and correctly diagnose compilation
failures in Triton-RISCV.

## Operator Semantics

GELU computes the Gaussian Error Linear Unit.

- `approximate="none"` uses the exact erf-based formula.
- `approximate="tanh"` uses the tanh approximation.

This task includes forward, in-place, backward, and dtype validation paths from
the existing test file.

## Reference Implementation

- PyTorch reference:
  - `torch.nn.functional.gelu(x, approximate=approximate)`
- Existing Triton or FlagGems reference:
  - `python/examples/flaggems/gelu.py`

## Input and Output Contract

- Inputs:
  - `x`: input tensor
  - `approximate`: GELU approximation mode, either `none` or `tanh`
- Outputs:
  - forward output tensor with the same shape as `x`
  - in-place output for `gelu_`
  - backward gradient output for `gelu_backward`
- Shapes:
  - `(512,)`, `(1023,)`, `(1024,)`
  - backward shapes `(64, 128)` and `(128, 128)`
- Dtypes:
  - `torch.float16`
  - `torch.float32`
  - `torch.float64`

## Files to Inspect

Codex should inspect these files before editing:

- `docs/codex-operator-workflow.md`
- `codex/templates/operator-task.md`
- `python/examples/flaggems/gelu.py`
- `python/examples/flaggems/test_gelu.py`
- `python/examples/flaggems/erf.py`
- `python/examples/flaggems/test_erf.py`

## Files to Modify

For the validation workflow, Codex should not modify implementation files unless
the failure is proven to be an operator implementation issue.

Allowed workflow files:

- `tasks/operators/gelu.md`
- `experiment-results/operator-validation.md`
- `docs/codex-operator-workflow.md`

Implementation files are out of scope for the current validation pass:

- `python/examples/flaggems/gelu.py`
- `python/examples/flaggems/test_gelu.py`

## Triton-RISCV Constraints

- Keep existing block-based elementwise patterns.
- Preserve both `none` and `tanh` approximation modes.
- Keep dtype conversions explicit.
- Do not introduce autotuning for this validation task.
- Do not modify compiler internals under `lib/`, `include/`, or `backend/`
  unless a separate compiler-fix task is created.

## Failure Triage Focus

The first validation attempt showed that `gelu` partially fails during
compilation:

- Result: 9 failed, 5 passed
- Log file: `test-logs/gelu-20260602-235012.log`
- Main error: `Dialect linalg not found for custom op linalg.generic`

Codex should classify this as an MLIR lowering or translation failure. The
failure is likely related to exact GELU or erf-like math lowering, because
standalone `erf` shows the same `linalg.generic` failure.

## Build and Test Commands

Set up the Triton-RISCV environment:

```sh
source scripts/triton-riscv-env.sh
```

Run the GELU test and save a log:

```sh
mkdir -p test-logs
log="test-logs/gelu-$(date +%Y%m%d-%H%M%S).log"
python -m pytest -q python/examples/flaggems/test_gelu.py -s 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "exit=$status" | tee -a "$log"
```

## Success Criteria

- Codex identifies the GELU approximation modes and test coverage.
- Codex distinguishes passing tanh-like paths from failing exact/erf-related
  paths.
- Codex records the failure stage as MLIR lowering or translation, not numerical
  correctness.
- Codex records the `linalg.generic` issue and links it to standalone `erf`
  validation.

## Validation Attempts

### Attempt 1: 2026-06-02

- Test command: `python -m pytest -q python/examples/flaggems/test_gelu.py -s`
- Failure stage: MLIR lowering or translation
- Compilation result: failed before runtime execution for several cases
- Test result: 9 failed, 5 passed
- Failure log: `test-logs/gelu-20260602-235012.log`
- Residual IR or dialect issue: generated MLIR contained `linalg.generic`
- Likely failure reason: exact GELU or related math lowering leaves linalg
  operations that `mlir-translate --mlir-to-llvmir` cannot handle
- Workflow improvement: add a failure triage step that checks for residual
  unsupported dialect operations before asking Codex to edit operator code
