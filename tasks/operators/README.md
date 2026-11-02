# Operator Task Files

This directory contains Codex task descriptions used by the first
Triton-RISCV operator workflow validation cycle.

## Current Task Roles

- `silu.md`: generated with `codex/generate_operator_task.py` and completed with
  the successful validation result. This is the passing reference task.
- `sigmoid_and_mul.md`: generated with `codex/generate_operator_task.py` for a
  new fused operator implementation. The implementation and test passed on the
  RISC-V validation server.
- `relu_and_mul.md`: generated with `codex/generate_operator_task.py` for a
  second new fused operator implementation. The implementation and test passed
  on the RISC-V validation server.
- `gelu.md`: validation and failure-triage task for GELU. This task records a
  mixed result and identifies an MLIR lowering or translation failure involving
  `linalg.generic`.
- `erf.md`: validation and failure-isolation task for standalone `tl.math.erf`.
  This task explains why GELU exact-mode failure is likely connected to math
  lowering rather than GELU wrapper logic.
- `vec_add.md`: baseline environment and compilation-flow task.

## Implementation Scope

The first validation cycle used existing repository operators and tests, then
added `sigmoid_and_mul` and `relu_and_mul` as small generated fused operators.
No compiler source file was modified. This was intentional: the observed GELU
and ERF failures happen before runtime numerical comparison and are classified
as compiler/lowering coverage issues.

Future cycles can use the same generator and template to create true
implementation tasks for missing operators. In that case, Codex should modify
only the implementation and test files listed in the generated task.
