# BOSCAME FPGA examples

This tree owns every Triton kernel and Python frontend driver in the
repository. There are no per-model README files: shape, grid and ABI live
in each case's `kernel.py` (`KERNEL_CONFIG`). The AOT API is
[docs/BoscameAOT.md](../../docs/BoscameAOT.md).

Export (`make ttir` / `tools/export.py`) needs the installed triton-riscv
plugin and `triton-shared-opt` (`TRITON_SHARED_OPT_PATH` or PATH). It does
not need Buddy. Later stages (plan, emit, image, simcheck) need an explicit
Buddy checkout:

```sh
source scripts/triton-riscv-env.sh
export BUDDY_ROOT=/path/to/buddy-mlir
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin
export LLVM_BIN=/path/to/llvm-build/bin
```

Buddy can also consume a handoff IR produced here:

```sh
make -C "$BUDDY_ROOT/examples/FPGA-BOSCAME/matmul_i8_196x768x768" \
  FRONTEND_IR=/path/to/export/kernel.frontend.mlir all check simcheck
```

## Models

| Directory | Role | Cases |
| --- | --- | ---: |
| `e5-small-v2` | E5-small encoder ops | 12 |
| `gte-small` | GTE-small encoder ops | 12 |
| `electra-base-discriminator` | ELECTRA-base ops | 15 |
| `xlnet-base-cased` | XLNet-base ops | 12 |
| `efficientnet_b3.ra2_in1k` | EfficientNet-B3 ops | 99 |
| `mobilenetv3_small_100.lamb_in1k` | MobileNetV3-Small ops | 88 |
| `vit-base-patch16-224` | ViT-B/16 ops plus graph tools | 40 |
| `layernorm_197x768` | Standalone LayerNorm `[197,768]` | 1 |
| `matmul_i8_128x1024x128` | Standalone i8 GEMM `128x1024x128` | 1 |
| `matmul_i8_196x768x768` | Standalone i8 GEMM `196x768x768` | 1 |

Shared helpers for every model live in that model's `tools/` and in
`examples/BOSCAMEFPGA/tools/`. Case names are the subdirectories that
contain `kernel.py`; `cases.py` lists them without importing Triton.

## Layout

A model with a case suite looks like:

```text
<model>/
    makefile          # suite: asm, simcheck, CASES=, KERNELS=
    cases.py          # KERNEL_CONFIG literals
    common.mk         # compiler stages
    <case>/
        kernel.py     # @triton.jit + KERNEL_CONFIG
        launch.c      # host / board harness
        makefile      # includes ../common.mk
```

Standalone examples omit `cases.py` and keep a single kernel (plus
`build.py` / `kernels.py` where needed).

## Common commands

From the repository root, after `source scripts/triton-riscv-env.sh`:

```sh
EX=examples/BOSCAMEFPGA

# One operator (works for suite cases and standalone trees):
make -C "$EX/e5-small-v2/add_128x384" PYTHON=python3
make -C "$EX/matmul_i8_196x768x768" ttir PYTHON=python3

# Whole model suite (continue on failure, record a report):
make -C "$EX/gte-small" asm PYTHON=python3
make -C "$EX/gte-small" simcheck PYTHON=python3
make -C "$EX/gte-small" asm CASES='add_128x384 gelu_128x1536'
```

Typical Make stages: `ttir` (export only), `boscame`, `llvm`, `asm`,
`check`, `simcheck`. Outputs under `build/` or `build/kernels/<case>/`
include `kernel.ttir`, `kernel.frontend.mlir` and
`contract-manifest.json`. Do not commit `build/`.

Integration tests are `test/Integration/BOSCAMEFPGA`. Enable them with
`-DBUDDY_ROOT=... -DBUDDY_MLIR_BINARY_DIR=...`; they are not part of the
frontend core lit suite.

## ViT graph (only this model)

`vit-base-patch16-224` is the only tree that also captures a complete
HuggingFace graph. The pinned checkpoint is
`google/vit-base-patch16-224` (batch 1, `[1,3,224,224]`, 196 patches plus
CLS, 12 blocks, hidden 768, 12 heads of 64, MLP 3072, 1000 logits).
`model/model.json` records the revision and checkpoint digest. The
checkpoint (~346 MB) is not in git; `tools/fetch_assets.py` verifies
SHA256.

Attention is QK, scale `1/sqrt(64)`, softmax, PV. Dense biases stay
separate. Exact GELU uses `erf`; LayerNorm uses `eps=1e-12`. The ordinary
graph has 245 numerical calls. W8A8 is the default AME path: each of 73
linears becomes quantize, signed i8 matmul (`matmul_i8_overwrite_*`,
`C=A*B` from a zero fill) and dequantize. `--fp32-reference` selects
float dense kernels; composed patch projection stays mixed precision.

```sh
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
export PYTHONPATH="$PWD/buddy-mlir/build/python_packages${PYTHONPATH:+:$PYTHONPATH}"

python "$VIT/tools/fetch_assets.py" --output "$VIT/assets"
python "$VIT/tools/import_model.py" --assets "$VIT/assets" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --output "$VIT/build/import" --save-mlir
make -C "$VIT" boscame PYTHON=python3
python "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels" --output "$VIT/build/replacement-w8a8" \
  --require-complete
python "$VIT/tools/lower_model.py" --replacement "$VIT/build/replacement-w8a8" \
  --output "$VIT/build/lowered"
python "$VIT/tools/model_simcheck.py" --mode w8a8 \
  --replacement "$VIT/build/replacement-w8a8"
```

`model_simcheck.py` is a prerequisite audit (exit 2 / `SKIP` if a required
kernel is missing). It does not claim a full-graph FPGA run. Accuracy
comparison uses `tools/compare_accuracy.py` against the pinned processor
and checkpoint hashes.
