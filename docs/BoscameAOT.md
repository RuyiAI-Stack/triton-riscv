# BOSCAME AOT frontend

`triton.backends.triton_shared.contract.export_contract` exports Triton ASTSource
or TTIR to verified standard buffer MLIR and a manifest. It accepts only
`GPUTarget("boscame", "generic", 1)` and an explicit three-dimensional positive
static `launch_grid`. CPUBackend remains the sole concrete backend plugin; it
dispatches BOSCAME export to an ordinary Python delegate. No active driver or
ordinary object cache is used. Normal `triton.compile()` rejects this target as
AOT-only.

```python
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.contract import export_contract

artifact = export_contract(
    "kernel.ttir",
    target=GPUTarget("boscame", "generic", 1),
    launch_grid=(2, 1, 1),
    entry_name="kernel",
    caller_contract="args_aligned64_nonalias",
)
artifact.write("handoff")
```

The ABI promise is optional and explicit. The target never implies alignment,
nonalias or approximation authorization. `math_accuracy` is an optional erf/exp
dictionary; each entry has exactly `domain="all-f32"`, `special_values="ieee"`,
`max_abs_error` and `max_rel_error` (finite nonnegative budgets). Unknown options
are rejected. Boolean frontend options are `debug`, `enable_make_gather_scatter`,
`enable_collapse_shape`, `enable_fp_fusion` and `allow_fp_reassoc`; the latter two
must be false. CPU/IME environment options do not affect this export.

Both AST and file paths run TTIR normalization, then explicit Contract-mode
lowering and bufferization. Input versions are checked before rewriting.
SymbolTable performs entry renaming and updates uses. A fresh parser re-verifies
the serialized artifact. Gluon, later-stage IR and remaining uninlined calls
are rejected. Verified incomplete graphs retain their limitation; a consumer
planner may reject them.

The returned `ContractArtifact` has `ttir`, `contract` and `manifest`. `.write()`
produces `kernel.ttir`, `kernel.frontend.mlir` and `contract-manifest.json`.
The manifest binds input/output hashes, target, entry ABI/grid, actual graph and
promises, options, Contract schema version, loaded Triton/Python components, frontend
tool/source identity and LLVM/MLIR build hashes. It omits developer absolute
paths. Each export compiles and verifies again; there is no persistent AOT cache.

Select the frontend with `TRITON_SHARED_OPT_PATH`, a bundled installed tool or
PATH. `triton-shared-opt --contract-build-info` reports its compiled identity.
The existing `--triton-to-linalg-experimental` CLI remains, with
`contract=false` by default. CPU, RISC-V, IME, AtomicCAS and FP8 compilation do
not enter the Contract pipeline by default.

Buddy's `docs/KernelContract.md` defines the normative neutral schema. This
repository owns its strict producer implementation in
`include/triton-shared/Analysis/KernelContract.h` and
`lib/Analysis/KernelContract.cpp`, built as part of `TritonSharedAnalysis`.
The build identity includes these local sources; there is no separate shared
package or source mirror. There is no Buddy SDK, header, library or tool
dependency for AOT export. BOSCAME planning,
recipes, emission, objects and firmware belong to the consumer. The frontier
contains no target dialect, target storage decisions or `plan.*` attributes.

Build this repository with its fixed `triton-hash.txt`, the complete patch stack
and a verified LLVM/MLIR build. `scripts/apply_patches.sh <triton-source>` is
idempotent; do not reset a checkout containing applied patches. For native
changes, run `make` in the Triton directory before testing. Then run
`check-triton-shared-lit-tests` in its build directory. Python lit tests require
NumPy and the installed frontend in the configured Python environment. API and
Contract conformance regression tests live in `test/Contract`.
Buddy's separate integration suite runs the actual two-process handoff and
strict target simulators; these simulators are not frontend core dependencies.

## Example ownership

All BOSCAME Triton Python kernels, the three standalone example drivers and the
complete ViT model live in [examples/BOSCAMEFPGA](../examples/BOSCAMEFPGA).
Export-only goals need no Buddy installation. Buddy consumes the resulting
`kernel.frontend.mlir`; image and simulator workflows use an explicit
`BUDDY_ROOT` and tool configuration. Cross-repository tests are under
`test/Integration/BOSCAMEFPGA` and are enabled separately from the core suite.
