# ViT-Base/Patch16-224 through neutral kernel contracts

Each real Triton kernel is exported by the external `triton-riscv` BOSCAME AOT
API to verified standard Contract IR plus a manifest. Buddy owns planning and
all subsequent stages. Set `TRITON_SHARED_OPT_PATH`, use explicit Buddy tool
variables or PATH, set `BUDDY_ROOT` for backend filters and simulators, and see the [handoff guide](../../../docs/BoscameAOT.md).

This ports `google/vit-base-patch16-224` from the upstream
`CBalaa/buddy-mlir-for-fpga` branch `vit-base-patch16-224`, commit
`478766cfddb11e61560c44d68320ee21279b90ae`. It retains all 34 fixed-shape
Triton specializations and the 12-block model capture/replacement sources,
with five additional overwrite matmuls and a weight-quantization check.

**A fresh complete-inventory build passes all 40 entries through restricted
target assembly and object generation.** All 418 kernel artifacts from the
previously supported 38 cases retain their SHA256 values. GELU and Softmax pass three
complete-grid strict-simulator rounds: 21276 programs and 3212676 output
comparisons. Their explicit approximation contracts retain the existing
`rtol=atol=2e-5`; the source algorithms remain unchanged. The emitted W8A8
classifier Linear chain passes four numerical stress rounds, and PV passes
three complete grids (1872 programs, 453888 bit-exact f32 outputs).
W8A8 dense linears are the default model path;
complete-model execution and accuracy remain unverified. See [VALIDATION.md](VALIDATION.md)
for the exact cases, diagnostics and numerical evidence. Nothing has been
executed on a board.

## Directory layout

Each operator is independent:

```text
add_197x768/
    kernel.py       # This operator's real @triton.jit body
    metadata.json   # Fixed logical shape and family
    launch.c        # Independent int launch(void), input generation and oracle
    build.py        # Standalone entry to the common compilation driver
    simcheck.py     # Standalone entry to the real emitted-IR simulator
    makefile
```

The same layout applies to all 40 cases. `cases.py` contains specialization
metadata, not shared kernel bodies. `tools/build.py` loads the selected local
`kernel.py` through Triton's actual `ASTSource`/BOSCAME AOT backend. There is no
monolithic kernel implementation or synthesized Linalg fallback.

The generated typed descriptor adapter follows the standalone
`../matmul_i8_128x1024x128` example: fixed shape/stride, 64-byte alignment,
nonalias/overflow checks, rank-one unranked descriptors, and a complete
three-dimensional program-grid loop. Offsets and lengths use elements.
Both i8 and f32 linear B operands have physical `[N,K]` storage.

`patch_embed_196x768/kernel.py` performs im2col. Its **public adapter** retains
the upstream composition: im2col, activation/weight quantization, i8 matmul,
dequantization, bias addition and NCHW output conversion. Compiling im2col
alone does not establish that the composed patch projection works. Its
complete execution is tracked separately from the emitted Linear chain.

## Build and simulate

Activate the Python environment used by the standalone Triton examples
(Python 3.12 with Triton/triton-riscv and NumPy), then run from the repository
root. The exporter locates the in-tree plugin; `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC` and `CLANG` can override individual tools.

```bash
export BUDDY_ROOT=/path/to/buddy-mlir
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin
export LLVM_BIN=/path/to/llvm-build/bin
export PATH="$BUDDY_MLIR_BINARY_DIR:$LLVM_BIN:$PATH"
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224

# One independently launchable operator:
make -C "$VIT/matmul_i8_197x768x768" PYTHON=python3
make -C "$VIT/matmul_i8_197x768x768" simcheck PYTHON=python3

# Attempt every operator, recording failures and continuing to the next:
make -C "$VIT" asm PYTHON=python3
make -C "$VIT" simcheck PYTHON=python3

# Reuse simulation results only after checking current hashes and full coverage:
make -C "$VIT" simcheck PYTHON=python3 COLLECT=1

# Actual emitted classifier quantize/dot/dequantize/bias/residual chain:
make -C "$VIT" w8a8-chaincheck PYTHON=python3

# Complete GELU/Softmax grids; one dirty-SPM Machine per shard across 3 rounds:
python3 "$VIT/tools/simcheck.py" --case gelu_197x3072 \
  --case softmax_12x197x197 --nonlinear-workers 48
# Independent 80-digit samples of the actual output archive (requires mpmath):
python3 "$VIT/tools/check_nonlinear_numerics.py"

# ABI-only smoke checks of the independent C launchers:
make -C "$VIT" launchcheck PYTHON=python3

# Explicit full-model prerequisite audit; currently SKIP, exit status 2:
make -C "$VIT" model-simcheck PYTHON=python3 MODE=w8a8
```

Suite commands attempt every selected case and return nonzero when a case is
blocked or fails; compilation and complete execution are recorded separately.
Per-case wrappers also accept the common driver's
`--stage ttir|boscame|llvm|asm`, `--output` and `--report` arguments.
Single-case summaries are written inside `build/kernels/<case>/`; an explicit
multi-case subset uses `build/kernels/*-subset-report.json`. These commands
preserve the complete-suite reports in `build/`. An explicit `--report` path
overrides the default destination.

Artifacts live in `build/kernels/<case>/`:

```text
kernel.ttir
  -> kernel.frontend.mlir  # Standard buffer-form contract, already bufferized
  -> kernel.bufferized.mlir # Separate kernel-contract-verify checkpoint
  -> kernel.planned.mlir   # bosc-ame-plan + bosc-ame-plan-verify
  -> kernel.boscame.mlir   # bosc-ame-emit
  -> kernel.llvm.mlir -> kernel.ll
  -> kernel.s -> kernel.encoded.s -> kernel.nr.s -> kernel.o
```

The `asm` stage includes the same fixed-register AME encoder and strict NR
assembly restriction used by the standalone example, followed by RISC-V
assembly to an object. It does not link a boot image or run the board.
The C oracle support is host-only; this port does not claim a board runtime.

`build/build-report.json`, `build/simcheck-report.json` and per-case result
files preserve the actual stage and diagnostic. Simulation executes emitted
BOSCAME operations in the strict DMA/SPM/REG interpreter, keeps poisoned SPM
between programs, checks guards/access ranges/unchanged inputs, and compares
every output. i8 cases run three complete grids, including signed extremes,
nonzero C, i32 wrap and tail-first/reverse scheduling. ABI smoke checks use
mock raw entries and explicitly do **not** validate numerical results.

## Model and numerical boundaries

[model/README.md](model/README.md) documents the pinned model assets and graph
capture/replacement commands. W8A8 is the default; `MODE=fp32` selects the
explicit dense floating-point reference. The patch composition remains W8A8. The model gate requires the current operator
artifacts and complete simulation evidence; host graph execution is not a
substitute for BOSCAME simulation. Full-model validation is skipped while required operators or assets are
missing. The new `tools/check_w8a8_chain.py` validates an emitted quantization
Linear chain independently; `tools/compare_accuracy.py` prepares the matched
checkpoint/input/label comparison without substituting host output for emitted
execution.

Two imported LayerNorm kernels contained an upstream padding defect: subtracting
the mean changed padded zero lanes to `-mean`, which then entered variance.
They now explicitly clear those lanes, matching the valid 768-column model
definition and the existing standalone LayerNorm example. This intentionally
changes the defective upstream results. Full-row validation against standard
LayerNorm and PyTorch is recorded in [VALIDATION.md](VALIDATION.md); it is not
claimed to be bit-identical to PyTorch's reduction tree.

Boundary copies now support planner-selected DMA/RVV segments or exact RVV
fallback, including the 1000-column cases. Each DMA still requires two
64-byte-aligned endpoints and a 64-byte-multiple length; RVV accesses only
the valid elements. Unused parent pitch does not constrain a single-row
copy. The simulator retains its access and DMA checks.

GELU's erf and Softmax's exp now use explicitly authorized FP32 polynomial
recipes; max reduction preserves its initializer, mask and NaN value semantics.
See the [nonlinear execution checkpoint](../../../docs/validation/nonlinear-operators/)
and [coefficient/error evidence](../../../docs/validation/nonlinear-math/).
The PV, nonlinear and W8A8 updates are recorded separately from historical
blockers in [VALIDATION.md](VALIDATION.md).
