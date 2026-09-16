# Operator Task: erf

## Objective

Use `erf` as a validation operator to isolate the math-lowering failure observed
in GELU exact mode.

## Operator Semantics

`erf` computes the elementwise Gaussian error function.

The existing implementation casts the input to `tl.float32` and stores
`tl.math.erf(x_f32)` to the output tensor. The task includes both normal and
in-place variants.

## Reference Implementation

- PyTorch reference:
  - `torch.erf(x)`
- Existing Triton or FlagGems reference:
  - `python/examples/flaggems/erf.py`

## Input and Output Contract

- Inputs:
  - `x`: input tensor
- Outputs:
  - `erf(x)` tensor with the same shape as `x`
  - in-place output for `erf_`
- Shapes:
  - `(512,)`
  - `(1023,)`
  - `(1024,)`
- Dtypes:
  - `torch.float32`
  - `torch.float16`
  - `torch.float64`

## Files to Inspect

Codex should inspect these files before editing:

- `docs/codex-operator-workflow.md`
- `codex/templates/operator-task.md`
- `python/examples/flaggems/erf.py`
- `python/examples/flaggems/test_erf.py`
- `python/examples/flaggems/gelu.py`
- `python/examples/flaggems/test_gelu.py`

## Files to Modify

For the validation workflow, Codex should not modify implementation files unless
the failure is proven to be an operator implementation issue.

Allowed workflow files:

- `tasks/operators/erf.md`
- `experiment-results/operator-validation.md`
- `docs/codex-operator-workflow.md`

Implementation files are out of scope for the current validation pass:

- `python/examples/flaggems/erf.py`
- `python/examples/flaggems/test_erf.py`

## Triton-RISCV Constraints

- Keep the existing block-based elementwise pattern.
- Keep masking explicit for tail elements.
- Keep dtype conversion to `tl.float32` before `tl.math.erf`.
- Do not introduce autotuning for this validation task.
- Do not modify compiler internals under `lib/`, `include/`, or `backend/`
  unless a separate compiler-fix task is created.

## Failure Triage Focus

The first validation attempt showed that all standalone erf cases fail during
compilation:

- Result: 18 failed
- Log file: `test-logs/erf-20260603-010841.log`
- Main error: `Dialect linalg not found for custom op linalg.generic`

Codex should classify this as an MLIR lowering or translation failure. This
result supports the GELU diagnosis because exact GELU depends on erf-like math
lowering.

## Build and Test Commands

Set up the Triton-RISCV environment:

```sh
source scripts/triton-riscv-env.sh
```

Run the ERF test and save a log:

```sh
mkdir -p test-logs
log="test-logs/erf-$(date +%Y%m%d-%H%M%S).log"
python -m pytest -q python/examples/flaggems/test_erf.py -s 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "exit=$status" | tee -a "$log"
```

## Success Criteria

- Codex identifies that `erf` uses `tl.math.erf`.
- Codex records the failure stage as MLIR lowering or translation, not numerical
  correctness.
- Codex records the `linalg.generic` issue.
- Codex links the `erf` failure back to the GELU exact-mode failure.

## Validation Attempts

### Attempt 1: 2026-06-03

- Test command: `python -m pytest -q python/examples/flaggems/test_erf.py -s`
- Failure stage: MLIR lowering or translation
- Compilation result: failed before runtime execution for all cases
- Test result: 18 failed
- Failure log: `test-logs/erf-20260603-010841.log`
- Residual IR or dialect issue: generated MLIR contained `linalg.generic`
- Likely failure reason: `tl.math.erf` lowering leaves linalg operations that
  `mlir-translate --mlir-to-llvmir` cannot handle
- Workflow improvement: use standalone math operators to isolate failures from
  fused or wrapper operator logic
