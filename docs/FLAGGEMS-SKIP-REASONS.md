# FlagGems CPU Backend Validation Status

This document records the current FlagGems validation status for the Triton
RISC-V CPU backend and the remaining global skip policy used by the example
test suite.

## Validation Command

Run the full example suite from the repository root with:

```sh
pytest python/examples/ \
  --ignore=python/examples/test_core.py \
  --ignore=python/examples/test_annotations.py \
  -v
```

The latest local validation result is:

```text
3892 passed, 280 warnings
```

The suite currently has no FlagGems file-level or parameter-level skips.

## Global Skip Policy

`python/examples/conftest.py` keeps only global skips for parameter values that
are not meaningful for the current CPU backend:

| Parameter class | Reason |
| --- | --- |
| `bfloat16` dtype parameters | bfloat16 linking issue |
| `tf32` input precision parameters | tf32 is not supported on CPU |
| `float8` dtype parameters | float8 is not supported on CPU |

These global guards do not hide any currently collected FlagGems test case in
the validation baseline above.

## Enabled Coverage

The current backend coverage includes the following areas that previously
needed backend-specific work or skip removal:

- Generic `tt.reduce` lowering, including multi-result reduction bodies used by
  variance and variance-mean kernels.
- Generic `tt.scan` lowering, including cumulative sum/product/min/max style
  kernels.
- Tensor-first structured-to-memref lowering for masked stores, split tensor
  pointers, and interleaved complex loads.
- Ordered CPU fallback lowering for `tts.atomic_rmw`, covering scatter, index,
  embedding, masked-select, and reduction-style update patterns.
- CPU reference paths for kernels whose CUDA/Triton implementation depends on
  unsupported target-specific features, such as convolution wrappers, attention
  wrappers, FP8 helpers, and stream-K matmul wrappers.
- Frontend math support for `tl.math.acos`, `tl.math.atan`, and
  `tl.math.atan2`, so `acos`, `atan`, `atan2`, and complex `angle` run through
  the normal Triton frontend path instead of local approximation helpers.

Any future skip growth should be treated as a regression against the validated
`3892 passed, 280 warnings` baseline unless it is tied to an explicitly
documented backend limitation.
