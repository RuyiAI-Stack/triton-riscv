# ELECTRA-base-discriminator / BOSCAME

This example follows `../vit-base-patch16-224` for
[`google/electra-base-discriminator`](https://hf-mirror.com/google/electra-base-discriminator).
Each standalone operator keeps `@triton.jit`, a literal `KERNEL_CONFIG`,
`int launch(void)` and a `makefile`, plus the typed MemRef to unranked
adapter, full grid dispatch, 64-byte alignment and non-alias checks, and the
NH/E6 runtime and linker scripts.

The current FPGA path defaults to scalar FP32 max lowering in the C++
pipeline. The complete model exports the embedding, 12 hidden layers and
logits: 14 tensors. After the scheduler download they are compared locally
with the emitted execution.

Acceptance runs the real C model schedule and the compiler's scalar LLVM
locally, then compares that result with the BINs the scheduler downloaded.
The host executes AME INT8/INT32 dots as exact C integer accumulation. The
board runs the normally compiled AME operators. Every output FP32 uint32 bit
pattern and every byte must match. One differing bit fails. This checks that
this compiled implementation matches on the FPGA.

```text
<operator>/         kernel.py + launch.c + makefile; 32 standalone operators
model/              pinned upstream config, revision and asset manifest
model_runtime/      complete C schedule, typed ABI and NH/E6 model entry
tools/              export, build, parameter prep, return and local checks
common.mk           shared AOT compile rules
board.mk/board.ld   standalone NH/E6 link rules and layout
board_model.ld      complete-model link layout
assets/             checkpoint and tokenizer (gitignored)
build/              build products, weights and returned BINs (gitignored)
```

The specialization is batch 1, sequence 128, eval, 12 layers, hidden and
embedding 768, 12 heads of 64, FFN 3072, vocabulary 30522, max positions 512
and LayerNorm epsilon `1e-12`. The output is raw per-token replacement
logits of shape `[128]`, including padding positions. Dropout is off at
inference.

Matrices use physical `B[N,K]` and overwrite `C=A@B.T`. W8A8 row
quantization is symmetric signed INT8, ties away from zero, with scale 1 on
an all-zero row. QK applies `0.125` once at dequantization. PV quantizes an
explicit transpose of V at runtime. Embedding, LayerNorm, bias, GELU,
softmax and the residual stay FP32. `cases.py` checks `BM<=M`, `BN<=N` and
`BK<=K`. QK uses BK=64. The final N=1 dot uses exact scalar INT32
accumulation. Other matmuls keep AME `tl.dot`. The standalone
`quantize_transpose_128x128` is an extra layout primitive and is not part of
the default model.

The prequantized board model runs 1,571 operator calls and returns the
embedding, 12 hidden layers and logits. The dynamic-weight C API
`electra_infer()` remains and makes 1,645 typed calls. Prequantized fixed
weights skip 74 of those weight quantizations. The caller supplies
64-byte-aligned non-aliasing tensors, legal INT32 token/type/mask values and
a 16 MiB workspace.

Build from the repository root into a separate output directory:

```sh
source scripts/triton-riscv-env.sh
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
ELECTRA="$PWD/examples/BOSCAMEFPGA/electra-base-discriminator"
OUT="$ELECTRA/build/example"

make -C "$ELECTRA" launchcheck KERNELS="$OUT/kernels"
make -C "$ELECTRA" model-abicheck
make -C "$ELECTRA" asm bin KERNELS="$OUT/kernels"
make -C "$ELECTRA" assets
python "$ELECTRA/tools/prepare_weights.py" --output "$OUT/weights"
python "$ELECTRA/tools/prepare_model_board.py" \
  --weights "$OUT/weights" --output "$OUT/model-board"
python "$ELECTRA/tools/check_model_weights.py" \
  --weights "$OUT/weights" --build "$OUT/model-board"
python "$ELECTRA/tools/build_model_bin.py" \
  --kernels "$OUT/kernels" --output "$OUT/model-board"
python "$ELECTRA/tools/build_model_host.py" \
  --kernels "$OUT/kernels" --build "$OUT/model-board" --output "$OUT/model-host"
```

`ttir`, `boscame`, `llvm` and `asm` are selectable stages. `CASES='...'`
selects a subset. Each operator directory builds an ELF and BIN by default.
`launchcheck` and `model-abicheck` check descriptors, the grid and the real
C schedule ABI. Numerical acceptance is the scheduler return compared with
the local emitted result. The pinned checkpoint revision is
`1ae76a97c7e84a4e640876a07453fccd636f0667`. Dtype, shape, offset, original
name and SHA of parameters and tokenizer inputs are written into the
manifest.

The shared FPGA policy defaults to `ALLOW_VECTOR_MAX_REDUCTIONS=0` and passes
`scalar-f32-max-reductions=true` to `lower-bosc-ame`. The C++ pass uses
bounded scratch, a volatile vector snapshot, per-lane scalar loads and an SSA
accumulator, and keeps the other vector operations. Changing the policy
rebuilds from LLVM onward. The assembly filter and the final ELF audit reject
a remaining `vfredmax.vs`. This policy is independent of the ordered-add
workaround. Non-FPGA compiler defaults stay unchanged. See
[the compiler policy note](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-max-reductions.md).

After the fpga112 forward and token are configured, run in the foreground:

```sh
export FPGA_SCHEDULER_CLIENT="$PWD/../fpga-scheduler/.venv/bin/fpga-scheduler"
export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"

# Standalone operators. Repeat --case to select a subset.
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model electra-base-discriminator --kernels "$OUT/kernels" --timeout 1800

# Complete model: run, download every BIN, check metadata and SHA, then
# accept locally.
python "$ELECTRA/tools/run_model_board.py" \
  --build "$OUT/model-board" --reference "$OUT/model-host" \
  --output "$OUT/model-run" --timeout 7200
```

`--client`, `--endpoint` and `--token-file` override the connection. The
runner does not open an SSH tunnel. The output directory must be new. The
board only computes and snapshots data. The runner downloads the real BINs,
checks a successful exit and the image and output metadata, then compares
numbers locally.

An existing download can be rechecked fully offline. Write the review into a
new directory:

```sh
OLD="$OUT/model-run"
python "$ELECTRA/tools/run_model_board.py" \
  --validate-only "$OLD" --reference "$OUT/model-host" \
  --output "$ELECTRA/build/offline-review-new"

# Or write only a new strict bitwise report.
python "$ELECTRA/tools/verify_model_emitted.py" \
  --run "$OLD" --reference "$OUT/model-host" \
  --output "$ELECTRA/build/offline-review-single.json"
```

`verify_model_emitted.py` refuses to overwrite an existing report and checks
the hash link among the task, image, parameters, reference sources and
objects. The runner returns 0 on success and nonzero when the numeric or
provenance check fails.

`board.ld` keeps the NH entry `0x80000000`, the E6 entry `0x80010000`, and
the NOLOAD/DMA/PMP convention. The full FP32 vocabulary is about 89.4 MiB.
The standalone arena is 112 MiB, the offline link region is 128 MiB, and the
E6 stack starts at `0x87800000`. The complete model uses `board_model.ld`.
Parameters, the 16 MiB scratch and output snapshots are allocated separately.
Git keeps sources, configuration and this note. `assets/`, `build/` and the
Python cache stay ignored.
