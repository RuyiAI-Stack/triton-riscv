# Operator Development Task: tanh_and_mul

## Immutable Contract

- Semantics: Broadcast x and y to a common shape, then compute tanh(x) multiplied elementwise by y.
- PyTorch reference: `torch.tanh(x) * y`
- Output: A tensor with the broadcast shape of x and y and values equal to torch.tanh(x) * y.
- Shape cases: (512,), (1023,), (1024,)
- Per-input shape cases: `[{"x": [4, 1], "y": [1, 8]}, {"x": [1, 17], "y": [3, 1]}]`
- Dtypes: torch.float32, torch.float16
- Maximum tolerance: rtol=0.01, atol=0.01
- Backward validation: required

### Inputs

- `x`: Floating-point input tensor passed through tanh.
- `y`: Floating-point tensor multiplied with tanh(x); broadcasting is allowed.

## Automatically Selected References

- `python/examples/flaggems/sigmoid_and_mul.py` (sigmoid_and_mul; tl ops: arange, constexpr, exp, float32, load, program_id, store)
- `python/examples/flaggems/relu_and_mul.py` (relu_and_mul; tl ops: arange, constexpr, float32, load, maximum, program_id, store)
- `python/examples/flaggems/gelu_and_mul.py` (gelu_and_mul; tl ops: arange, constexpr, erf, exp, float32, load, program_id, store)
- `python/examples/flaggems/silu_and_mul.py` (silu_and_mul; tl ops: arange, constexpr, exp, fdiv, float32, load, program_id, store)

## Allowed Files

- Implementation: `python/examples/flaggems/tanh_and_mul.py`
- Test: `python/examples/flaggems/test_tanh_and_mul.py`

Do not modify the contract, reference expression, shape/dtype coverage, or
tolerances merely to make validation pass. During repair iterations, the test
file is locked and only the implementation may be changed.

## Required Work

1. Inspect the selected references and `docs/05-Operator Migration.md`.
2. Implement a clear Triton kernel and Python wrapper matching the contract.
3. Add pytest coverage that computes the PyTorch reference independently and
   compares it with `torch.testing.assert_close`.
4. Include every required shape and dtype and the backward path when required.
5. Keep pointer arithmetic, masks, casts, and boundary behavior explicit.
6. Do not use autotuning in the first implementation.

## Validation

```sh
source scripts/triton-riscv-env.sh
python -m pytest -q python/examples/flaggems/test_tanh_and_mul.py -s
```

The orchestration agent runs this command in the configured RISC-V environment,
classifies failures, and permits only bounded semantic-preserving repairs.

## Notes

Start with a simple one-dimensional block-based implementation and explicit masks.

<!-- autonomous-validation:start -->

## Autonomous Validation Record

- Final status: `passed`
- Acceptance-test SHA-256: `90fbf65c1a4cd0847794b43397eea0b1e24e63551a413044e906849db3eb2552`
- Repair attempts: 1

| Attempt | Status | Failure stage | Test summary |
| --- | --- | --- | --- |
| 1 | failed | mlir-translate | 12 failed in 26.88s |
| 2 | passed |  | 12 passed in 30.86s |
| strengthened coverage | passed |  | 20 passed in 16.79s |

### Failure Evidence

- `error: Dialect linalg not found for custom op linalg.generic`

The acceptance test remained locked during repair. The final test added two
broadcast shape combinations and reran forward and backward coverage for
`torch.float32` and `torch.float16`.

Full logs, Codex transcripts, and per-iteration patches are stored under
`agent-results/development/` in the validation workspace.

<!-- autonomous-validation:end -->
