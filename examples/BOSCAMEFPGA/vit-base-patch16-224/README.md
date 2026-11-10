# ViT-Base/Patch16-224 through neutral kernel contracts

Each real Triton kernel is exported by the external `triton-riscv` BOSCAME AOT
API to verified standard Contract IR plus a manifest. Buddy owns planning and
all subsequent stages. Set `TRITON_SHARED_OPT_PATH`, use explicit Buddy tool
variables or PATH, set `BUDDY_ROOT` for backend filters and simulators, and see the [handoff guide](../../../docs/BoscameAOT.md).

This ports `google/vit-base-patch16-224` from upstream
`CBalaa/buddy-mlir-for-fpga`, commit
`478766cfddb11e61560c44d68320ee21279b90ae`. Production matrix multiplication
uses quantization, signed i8 AME matmul (overwrite) and dequantization for patch
projection, every dense Linear, and QK/PV attention. The five standalone FP32
Linear entries are retired. The current inventory has 43 entries, including
10 new attention/layout primitives and the retained integer matrix examples.

QK/PV retain their model f32 shapes and layouts. Model replacement expands
each batch matmul into explicit two-dimensional primitive calls: left
quantization, runtime right quantization with transpose, overwrite i8 matmul,
and dequantization. Each of the 12 heads has four calls. Compiled pack/unpack
primitives stage the 197×197 heads at 64-byte-aligned addresses. There are no
standalone QK/PV cases or attention composition configurations.
The project naming convention is `gemm_i8_*` for `C = old C + A @ B.T`
and `matmul_i8_*` for `C = A @ B.T`. Both use physical `B[N,K]` storage;
production quantized chains use the overwrite `matmul_i8_*` family.
QK applies 0.125 once in `dequantize_197x197`; PV applies no extra factor.
[OPERATOR_NAMING_MAP.json](OPERATOR_NAMING_MAP.json) gives the machine-readable
old-to-new map. Complete-model execution and accuracy remain unverified.
New operator/chain success does not establish complete-model execution or accuracy.

The current cleanup prepares the complete standalone inventory for later
operator verification. It checks inventory, dependencies, references, Python
syntax and diffs only; it establishes no new numerical or FPGA result. The
commands below document the retained workflows for subsequent work.

## Directory layout

Each operator is independent:

```text
add_197x768/
    kernel.py       # Real @triton.jit body and literal KERNEL_CONFIG
    launch.c        # Independent int launch(void), input generation and oracle
    makefile        # Compiler stages plus the default NH/E6 bin target
```

The same layout applies to all 43 entries. Each `kernel.py` owns its shape,
arguments, constexprs (including tile sizes), launch grid and numerical
contracts in `KERNEL_CONFIG`. Keep this dictionary a Python literal:
`cases.py` reads it with `ast.literal_eval`, so enumerating cases needs neither
Triton nor PyTorch. `tools/export.py` loads the selected local kernel through
Triton's actual `ASTSource`/BOSCAME AOT backend and generates the ABI adapter.
`common.mk` owns the external compiler commands and file dependencies;
`tools/report.py` records and collects their results.
Each case's `make simcheck` target builds its BOSCAME module and calls the
shared `tools/simcheck.py` directly.

The generated typed descriptor adapter follows the standalone
`../matmul_i8_128x1024x128` example: fixed shape/stride, 64-byte alignment,
nonalias/overflow checks, rank-one unranked descriptors, and a complete
three-dimensional program-grid loop. Offsets and lengths use elements.
Both i8 families' B operands have physical `[N,K]` storage for `A @ B^T`.

`patch_embed_196x768/kernel.py` performs im2col. Its **public adapter** retains
the upstream composition: im2col, activation/weight quantization, i8 matmul,
dequantization, bias addition and NCHW output conversion. Compiling im2col
alone does not establish that the composed patch projection works. Its
complete execution is tracked separately from the emitted Linear chain.

## Build and simulate

FPGA builds default to sequential scalar accumulation for ordered f32 sums
in SSA. The final ELF audit rejects a remaining `vfredosum.vs`. Hardware
diagnosis uses `make ALLOW_VECTOR_ORDERED_REDUCTIONS=1`. Restoring `=0` does
not need `clean`. See
[the bypass note](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-ordered-reductions.md).

Activate the `boscame` conda environment configured in the repository README,
then run from the repository root. The exporter locates the in-tree plugin; `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC` and `CLANG` can override individual tools.

```bash
conda activate boscame
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224

# One independently launchable operator:
make -C "$VIT/gemm_i8_197x768x768" PYTHON=python3
make -C "$VIT/gemm_i8_197x768x768" simcheck PYTHON=python3

# Attempt every operator, recording failures and continuing to the next:
make -C "$VIT" asm PYTHON=python3
make -C "$VIT" simcheck PYTHON=python3

# Select cases or redirect all generated kernel artifacts:
make -C "$VIT" asm CASES='add_197x768 layernorm_1x768'
make -C "$VIT" -j4 asm KERNELS="$PWD/.cache/vit-kernels"

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
Use Make targets `ttir|boscame|llvm|asm`, `KERNELS=/absolute/output/path` and
`REPORT=/absolute/report.json` for compilation. Repeating an unchanged build
checks frontend/command identity and reports cached results without running
the exporter or compilers. Changes to a kernel rebuild that case; compiler
tools and flags also participate in dependency tracking. Failed rebuilds
invalidate the previous successful result before modifying artifacts.
Single-case summaries are written inside `build/kernels/<case>/`; an explicit
multi-case subset uses `build/kernels/*-subset-report.json`. These commands
preserve the complete-suite reports in `build/`. An explicit `--report` path
overrides the default destination (`REPORT=` for Make; `--report` for simulator
tools). Use `make simcheck` in a case directory for independent simulation;
the shared `tools/simcheck.py` accepts advanced simulator CLI arguments.

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
The explicit `bin` target links the NH/E6 board runtime described below.
The `asm` stage alone does not execute or validate an operator.

## Build and run on FPGA

Use the `boscame` conda environment and this repository's `buddy-mlir` checkout.
Each operator's default Make target is `bin`; compiler and simulator targets
remain available explicitly. The accepted `add_1x768` example retains its
standalone board launcher and linker script. Other examples share `board.mk`,
`board.ld` and `tools/board_support.c`, using the in-tree Buddy NH/E6 boot,
PMP, cache, UART and DMA support.

```bash
conda activate boscame
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224

make -C "$VIT/add_197x768"
python examples/BOSCAMEFPGA/run_bin.py \
  "$VIT/build/kernels/add_197x768/board/add_197x768.bin"

# Avoid recomputing an expensive scalar reference on the FPGA.
make -C "$VIT/gemm_i8_197x768x768" bin GOLDEN=1
python examples/BOSCAMEFPGA/run_bin.py \
  "$VIT/build/kernels/gemm_i8_197x768x768/board/gemm_i8_197x768x768.bin"

# Submit existing images one at a time and preserve evidence.
python "$VIT/tools/run_board_suite.py" \
  --case add_197x768 --report-dir "$VIT/build/board-runs/manual"
```

The user-managed SSH tunnel is `ssh -N -L 18765:localhost:18765 fpga113`.
`run_bin.py` submits to its scheduler, which queues jobs and executes one
operator at a time. The current execution deadline is 120 seconds, including
image load and CPU reset, with queue wait excluded. A deadline with UART
output is reported as `TIMEOUT`; a deadline with no UART is `HANG`.

`GOLDEN=1` is supported for integer matmuls and
the composed patch embedding. `tools/board_golden.py` evaluates the independent
`launch.c` expected algorithm on the host, retaining input initialization and
tolerance. The retained attention reference helpers distinguish quantized
arithmetic from the independent host FP32 baseline. The board runs the real
typed adapter and compares every output against the embedded reference.
`golden_manifest.json` records provenance, coverage and hashes. These host
reference checks alone do not establish board correctness. `GOLDEN=0`
(the default) keeps the scalar oracle on the board.

The reported `kernel_cycles` uses the E6 cycle counter around the complete
typed adapter, including grid dispatch, DMA and the final fence. It excludes
input initialization, numerical validation and UART. A board `PASS` requires
the expected invocations to return, numerical validation to complete, and
zero errors. A timeout without a kernel return has no completed cycle count.
The suite preserves each submitted binary and SHA256, full console and UART
logs, scheduler task details, ELF audit and per-case result JSON under the
selected report directory.

`build/build-report.json`, `build/simcheck-report.json` and per-case result
files preserve the actual stage and diagnostic. Compiler stdout/stderr is
retained in per-stage `*.log` files beside the artifacts. Simulation executes
emitted BOSCAME operations in the strict DMA/SPM/REG interpreter, keeps poisoned SPM
between programs, checks guards/access ranges/unchanged inputs, and compares
every output. i8 cases run three complete grids, including signed extremes,
nonzero C, i32 wrap and tail-first/reverse scheduling. ABI smoke checks use
mock raw entries and explicitly do **not** validate numerical results.

## Model and numerical boundaries

Historical board success is not counted as execution of a newly named image.
`make operatorcheck KERNELS=/absolute/evidence/kernels REPORT_DIR=/absolute/evidence`
rechecks saved evidence without building, simulating or submitting board jobs.

FPGA production supports only W8A8 matrix
products, including QK/PV. `MODE=fp32` and FPGA reference replacement are
disabled; `compare_accuracy.py reference` retains the independent host FP32
baseline.

```sh
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
export PYTHONPATH="$PWD/buddy-mlir/build/python_packages${PYTHONPATH:+:$PYTHONPATH}"
python "$VIT/tools/fetch_assets.py" --output "$VIT/assets"
python "$VIT/tools/import_model.py" --assets "$VIT/assets" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --output "$VIT/build/import" --save-mlir
python "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels" --output "$VIT/build/replacement-w8a8" \
  --require-complete
python "$VIT/tools/lower_model.py" --replacement "$VIT/build/replacement-w8a8" \
  --output "$VIT/build/lowered"
python "$VIT/tools/model_simcheck.py" --mode w8a8 \
  --replacement "$VIT/build/replacement-w8a8"
```

`model_simcheck.py` is a prerequisite audit and exits 2 when a required kernel
is missing. The model gate requires the current operator
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
LayerNorm and PyTorch is not claimed to be bit-identical to PyTorch's reduction tree.

Boundary copies now support planner-selected DMA/RVV segments or exact RVV
fallback, including the 1000-column cases. Each DMA still requires two
64-byte-aligned endpoints and a 64-byte-multiple length; RVV accesses only
the valid elements. Unused parent pitch does not constrain a single-row
copy. The simulator retains its access and DMA checks.

GELU's erf and Softmax's exp now use explicitly authorized FP32 polynomial
recipes; max reduction preserves its initializer, mask and NaN value semantics.
