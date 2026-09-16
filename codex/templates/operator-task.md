# Operator Task: {{OPERATOR_NAME}}

## Objective

Implement or complete the Triton-RISCV operator `{{OPERATOR_NAME}}` using the
workflow in `docs/codex-operator-workflow.md`.

## Operator Semantics

{{OPERATOR_SEMANTICS}}

## Reference Implementation

- PyTorch reference:
  - `{{PYTORCH_REFERENCE}}`
- Existing Triton or FlagGems reference:
  - `{{TRITON_REFERENCE}}`

## Input and Output Contract

- Inputs:
  - `{{INPUTS}}`
- Outputs:
  - `{{OUTPUTS}}`
- Shapes:
  - `{{SHAPES}}`
- Dtypes:
  - `{{DTYPES}}`

## Files to Inspect

Codex should inspect these files before editing code:

- `README.md`
- `docs/05-Operator Migration.md`
- `python/examples/`
- `python/examples/flaggems/`
- `{{ADDITIONAL_FILES_TO_INSPECT}}`

## Files to Modify

Codex may modify or add only the files needed for this operator:

- `{{IMPLEMENTATION_FILE}}`
- `{{TEST_FILE}}`

If other files appear necessary, explain why before modifying them.

## Triton-RISCV Constraints

Avoid unsupported or risky features unless the existing repository already uses
them successfully for a similar operator.

- Avoid autotuning in the initial implementation.
- Prefer simple block-based elementwise kernels for first validation.
- Compare against existing examples before introducing new Triton patterns.
- Keep pointer arithmetic, masking, and dtype conversions explicit.

## Environment Prerequisite Check

Before running build or test commands, check that:

- `../triton` exists, or `TRITON_DIR` points to a valid Triton checkout
- `../buddy-mlir` exists, or `BUDDY_DIR` points to a valid Buddy MLIR checkout
- `.venv` or the configured Python environment is available
- `pytest` is available in the active environment
- `scripts/triton-riscv-env.sh` can be sourced without path errors
- the host architecture is supported by the selected build path

If any item is missing, do not mark the operator as failed. Record an
environment setup failure in the validation log.

## Implementation Steps

1. Read the reference implementation and existing nearby examples.
2. Run the environment prerequisite check.
3. Write or complete the Triton kernel.
4. Add a Python wrapper if needed.
5. Add or update pytest coverage.
6. Compare results against the PyTorch reference implementation.
7. Run the requested test command.
8. Record the result in the validation log.

## Failure Triage Steps

If the test fails, classify the failure before modifying code:

1. Environment failure: missing Triton checkout, Buddy checkout, Python
   environment, pytest, or unsupported host architecture.
2. Import failure: Python cannot import the implementation or test module.
3. Triton compilation failure: JIT starts, but compilation fails before runtime.
4. MLIR lowering or translation failure: generated MLIR cannot be translated to
   LLVM IR.
5. Runtime failure: compiled code launches but crashes or returns invalid data.
6. Correctness failure: execution completes, but output differs from the PyTorch
   reference.

For MLIR lowering or translation failures, search the log for unsupported
dialect operations. If the log contains `linalg.generic` or `Dialect linalg not
found`, record the failure as a compiler/lowering coverage issue unless there is
clear evidence that the operator implementation generated invalid semantics.

Do not modify compiler internals or unrelated operator code as part of this task
unless the task explicitly asks for a compiler fix.

## Build and Test Commands

Set up the environment:

```sh
source scripts/triton-riscv-env.sh
```

Run the relevant test:

```sh
{{TEST_COMMAND}}
```

If the implementation requires a rebuild, run:

```sh
scripts/rebuild-triton-riscv.sh
```

## Success Criteria

- The operator implementation matches the stated semantics.
- The test compares against a PyTorch reference implementation.
- The relevant pytest command passes.
- Any unsupported cases are documented.

## Validation Attempts

Append a new attempt entry after each validation run. Do not replace older
attempts.

### Attempt 1: TODO

- Operator:
- Task file:
- Implementation files:
- Test command:
- Failure stage:
- Compilation result:
- Test result:
- Failure log:
- Residual IR or dialect issue:
- Likely failure reason:
- Workflow improvement:
