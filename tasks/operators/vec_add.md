# Operator Task: vec_add

## Objective

Use vec_add as a simple validation operator to check whether Codex can understand the basic Triton-RISCV elementwise kernel pattern, identify the PyTorch reference comparison, and run a minimal operator validation flow.

## Operator Semantics

vec_add computes the elementwise sum of two input tensors.

For each element, it computes `output = x + y`.

This validation task includes the forward kernel only.

## Reference Implementation

- PyTorch reference:
  - `x + y`
  - The reference comparison is in `python/examples/test_vec_add.py`.
- Existing Triton reference:
  - `python/examples/test_vec_add.py`

## Input and Output Contract

- Inputs:
  - `x`: input tensor
  - `y`: input tensor with the same shape as `x`
- Outputs:
  - `output`: tensor with the same shape as `x` and `y`
- Shapes:
  - one-dimensional tensors are used by the existing example
  - the example uses a large vector size for validation and benchmarking
- Dtypes:
  - `torch.float32` in the benchmark path

## Files to Inspect

Codex should inspect these files before editing:

- `docs/codex-operator-workflow.md`
- `codex/templates/operator-task.md`
- `docs/05-Operator Migration.md`
- `python/examples/test_vec_add.py`
- `README.md`

## Files to Modify

Codex should not modify the implementation for the first validation pass unless it finds a clear issue.

Allowed files for this validation task:

- `tasks/operators/vec_add.md`
- `python/examples/test_vec_add.py`

## Triton-RISCV Constraints

- Use the existing block-based elementwise kernel pattern.
- Keep `BLOCK_SIZE` explicit.
- Use `tl.arange`, `tl.load`, `tl.store`, and `mask` for boundary handling.
- Do not introduce autotuning for this validation task.
- Do not modify compiler internals under `lib/` or `include/`.

## Environment Prerequisite Check

Before running the test command, check that `../triton` and `../buddy-mlir` exist or that `TRITON_DIR` and `BUDDY_DIR` point to valid checkouts. Also check that the Python environment provides `pytest` and that `scripts/triton-riscv-env.sh` can be sourced without path errors. Confirm that the host architecture is supported by the selected build path.

If these prerequisites are missing, record an environment setup failure instead of marking `vec_add` as an operator implementation failure.

## Implementation Steps

1. Read `python/examples/test_vec_add.py`.
2. Run the environment prerequisite check.
3. Identify the forward formula `output = x + y`.
4. Confirm that the PyTorch reference is `x + y`.
5. Confirm that the kernel uses block offsets and mask-based boundary handling.
6. Run or report the relevant pytest command.
7. Record the validation result and any workflow issue found.

## Build and Test Commands

Set up the Triton-RISCV environment by sourcing `scripts/triton-riscv-env.sh`.

Run the vector add test with `pytest python/examples/test_vec_add.py -v`.

If the local build is stale or the operator does not compile, rebuild Triton-RISCV with `scripts/rebuild-triton-riscv.sh` and then run the pytest command again.

## Success Criteria

- Codex correctly identifies the operator formula.
- Codex identifies the block-based Triton kernel structure.
- Codex identifies the PyTorch reference comparison.
- The relevant pytest command passes.
- Any failure is recorded with the failing command and likely reason.

## Validation Attempts

Append a new attempt entry after each validation run. Do not replace older attempts.

### Attempt 1: 2026-06-02

- Test command: `python -m pytest -q python/examples/test_vec_add.py -s`
- Compilation result: passed through Triton-RISCV JIT compilation and execution
- Test result: 1 passed
- Failure log: none
- Likely failure reason: not applicable
- Workflow improvement: keep `vec_add` as the baseline check for environment,
  backend registration, compilation, execution, and PyTorch reference comparison
