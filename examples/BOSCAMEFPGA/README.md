# BOSCAME FPGA examples

This repository owns all Triton kernels and Python frontend drivers. Each
independent example exports TTIR, verified standard MLIR Contract and a manifest:

```sh
make -C examples/BOSCAMEFPGA/matmul_i8_196x768x768 ttir PYTHON=python3
```

The same export works for `matmul_i8_128x1024x128` and `layernorm_197x768`.
It requires the installed triton-riscv Python backend and `triton-shared-opt`,
selected with `TRITON_SHARED_OPT_PATH` or PATH. It requires no Buddy checkout.
Outputs are `build/<example>/kernel.ttir`, `kernel.frontend.mlir` and
`contract-manifest.json`. `build.py --export-contract --output DIR` selects
a different build root. See [the AOT API](../../docs/BoscameAOT.md).

Buddy's corresponding examples accept the handoff through `FRONTEND_IR`:

```sh
make -C "$BUDDY_ROOT/examples/FPGA-BOSCAME/matmul_i8_196x768x768" \
  FRONTEND_IR=/path/to/export/kernel.frontend.mlir all check simcheck
```

For convenience, the three frontend examples retain their complete image and
simulator workflows. For backend goals set `BUDDY_ROOT` to an external Buddy
checkout, `BUDDY_MLIR_BINARY_DIR` to its binary directory and `LLVM_BIN` to the
shared LLVM tools. No path is inferred from the checkout nesting.

`vit-base-patch16-224` contains the complete model, 40 independent kernel cases,
graph tools and numerical checks. Run its `ttir` goal without Buddy; subsequent
stages use explicit Buddy tools, and assembly filters/simulators use `BUDDY_ROOT`.
Its source algorithms, approximation authorizations and tolerances are preserved.

Optional cross-repository tests live in `test/Integration/BOSCAMEFPGA`. Configure
the frontend build with `-DBUDDY_ROOT=... -DBUDDY_MLIR_BINARY_DIR=...` to enable
`check-triton-contract-integration`. They are excluded from the frontend core
suite and use Buddy's strict simulator through the explicit checkout path.
