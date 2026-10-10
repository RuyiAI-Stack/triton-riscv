# Codex Task Generation Flow

This directory contains the simple task generation flow used by the
Triton-RISCV operator workflow.

## Purpose

The generator turns an operator name and a small amount of context into a Codex
task description file. The generated task tells Codex what to inspect, what it
may modify, how to run validation, and how to record success or failure.

## Generate a Task

Run from the repository root:

```sh
python codex/generate_operator_task.py <operator>
```

Example:

```sh
python codex/generate_operator_task.py silu \
  --semantics "SiLU computes x / (1 + exp(-x))." \
  --pytorch-reference "torch.nn.functional.silu(x)" \
  --triton-reference "python/examples/flaggems/silu.py" \
  --inputs "x: input tensor" \
  --outputs "output tensor with the same shape as x" \
  --shapes "512, 1023, 1024" \
  --dtypes "torch.float32, torch.float16, torch.float64" \
  --implementation-file "python/examples/flaggems/silu.py" \
  --test-file "python/examples/flaggems/test_silu.py" \
  --test-command "python -m pytest -q python/examples/flaggems/test_silu.py -s"
```

By default, the generated file is written to:

```text
tasks/operators/<operator>.md
```

Use `--force` to overwrite an existing generated task.

## Use the Generated Task

1. Give the generated task file to Codex as the operator context.
2. Ask Codex to inspect the listed implementation and test files.
3. Ask Codex to implement, complete, or validate the operator step by step.
4. Run the listed test command in a Triton-RISCV environment.
5. Append the result to `experiment-results/operator-validation.md`.
6. If a failure is found, update `docs/codex-operator-workflow.md` with the
   missing diagnostic rule.

## Current Validation Targets

The first validation cycle used:

- `silu`: passing elementwise activation validation.
- `gelu`: mixed result; exact/erf-related paths expose a lowering issue.
- `erf`: standalone failure-isolation case for `tl.math.erf` lowering.
- `sigmoid_and_mul`: generated fused operator implementation that passed
  forward and backward validation.
- `relu_and_mul`: second generated fused operator implementation that passed
  forward and backward validation.

The baseline environment test used `vec_add`.
