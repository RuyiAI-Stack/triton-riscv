# thenlper/gte-small — BOSCAME FPGA

This example follows `vit-base-patch16-224` for its Triton kernels, C
`int launch(void)` launchers, typed memref adapters, NH/E6 runtime and linker
scripts. It contains 30 independent operators and a complete 12-layer C model
with 1,649 operator calls.

## Model and operator contracts

The specialization is batch 1, sequence 128, FP32 activations, hidden width
384, 12 layers, 12 attention heads of width 32, and FFN width 1,536. Token IDs,
positions, token types and attention masks are INT32 `[128]`. Masks contain
0/1 and at least one valid token. Dropout is disabled. The output is the
attention-mask-weighted mean over valid tokens, including CLS/SEP, followed
by L2 normalization. GTE requires no text prefix.

Each operator directory provides `kernel.py`, `launch.c` and `makefile`.
`KERNEL_CONFIG` records physical shapes, output arguments, grid and numerical
contracts. Linear weights use `[output,input]` storage. INT8 matrix products
overwrite INT32 output; row quantization is symmetric, with FP32 scales and
ties away from zero. QK dequantization applies `1/sqrt(32)` once; PV applies
no additional factor.

All matmul tiles satisfy `BM<=M`, `BN<=N`, `BK<=K`. QK `128x128x32` uses an
exact INT32 multiply/add fallback (`BM=BN=1,BK=32`) because AME requires K
to be divisible by 64. The other four shapes use `tl.dot` and AME; PV
`128x32x128` uses `BN=32`.

## Build and local checks

From the repository root, using the ViT toolchain environment:

```bash
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
GTE=examples/BOSCAMEFPGA/gte-small
make -C "$GTE" hostcheck       # kernel contracts and typed C ABI
make -C "$GTE" runtimecheck    # routing, shapes, alignment and all 1,649 calls
make -C "$GTE" asm
make -C "$GTE" bin KERNELS="$PWD/$GTE/build/kernels"
make -C "$GTE" simcheck
```

`bin` builds and audits NH/E6 ELF/BIN files locally. It does not submit a
board task. Select operators with `CASES='add_128x384 layernorm_128x384'`, or
build one with `make -C "$GTE/embedding_lookup_128x384" bin`. Intermediate
goals are `ttir`, `boscame`, `llvm` and `asm`.

FPGA builds enable the compiler's scalar FP32 ordered-add and max reduction
lowering by default. Other vector arithmetic remains enabled. The final ELF
audit rejects remaining vector reductions covered by those policies. See
[the max lowering documentation](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-max-reductions.md).
This is a software workaround; the hardware cause remains undetermined.

## Complete model and strict comparison

Fetch the pinned checkpoint and build a new output directory:

```bash
make -C "$GTE" assets
python "$GTE/tools/build_model_bin.py" --assets "$GTE/assets" \
  --kernels "$GTE/build/kernels" --output "$GTE/build/full-model"
python "$GTE/tools/build_model_host.py" --build "$GTE/build/full-model" \
  --kernels "$GTE/build/kernels" --output "$GTE/build/full-model-host-emitted"
```

The builder packs the real checkpoint and tokenizer inputs and records the
static operator sequence. The pinned revision is
`17e1f347d17fe144873b1201da91788898c639cd`; `tools/fetch_assets.py` records
asset SHA256 hashes. Weights stored as FP16 are converted to FP32 before
binding C pointers. `model/` retains configuration and provenance; tokenizer
execution stays on the host.

The local execution uses the actual C model graph, launcher, typed adapters
and emitted scalar LLVM. Matrix products use exact signed INT8/INT32 C dot
products. Its manifest binds inputs, parameters, graph, kernel artifacts and
result BINs by SHA256. This local run supplies comparison data, while FPGA
execution evidence comes from scheduler-downloaded BINs.

For an authorized future board session, the normal runner freezes the matching
local outputs before submission, keeps the scheduler client in the foreground,
downloads all declared output BINs and verifies metadata, byte counts and
hashes before comparing every byte:

```bash
python "$GTE/tools/run_model_board.py" --build "$GTE/build/full-model" \
  --kernels "$GTE/build/kernels" \
  --emitted-reference "$GTE/build/full-model-host-emitted" \
  --output "$GTE/build/full-model-board/new-run" --timeout 7200
```

Connection settings accept `--client`, `--endpoint`, `--token-file` or the
`FPGA_SCHEDULER_CLIENT`, `FPGA_SCHEDULER_URL`, `FPGA_SCHEDULER_TOKEN_FILE`
environment variables. The client defaults to the sibling scheduler checkout
or PATH; the endpoint defaults to local port 18762.
Board code only computes and captures outputs. All numerical verification runs
locally. Missing or stale emitted references fail before submission; any bit
difference is a failure, including 1 ULP. `--capture-only` is reserved for
separate diagnostic captures.

An existing capture can be verified offline without accessing the scheduler:

```bash
python "$GTE/tools/run_model_board.py" \
  --validate-only "$GTE/build/full-model-board/new-run" \
  --emitted-reference "$GTE/build/full-model-host-emitted" \
  --kernels "$GTE/build/kernels" \
  --validation-output "$GTE/build/offline-review/gte-emitted.json"
```

Choose a new report path for each review. Quantization operators export both
INT8 Q and FP32 scales. To run independent operators with scheduler download
and local C launcher validation:

```bash
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py --model gte-small \
  --kernels "$GTE/build/kernels" --output "$GTE/build/operator-runs" --timeout 1800
```

## Runtime and known hardware differences

`model_runtime/model.h` exposes `gte_workspace_bytes()` and `gte_infer()`.
The caller supplies checkpoint pointers, aligned inputs, a 7,479,296-byte
workspace, sentence output and optional last hidden output. The runtime checks
input values, workspace size and nonoverlap before dispatch. Buffers are
contiguous and aligned to 64 bytes.

`board.ld` retains NH entry `0x80000000` and E6 entry `0x80010000`, with NOLOAD
snapshots, explicit DMA/PMP bounds, a 64 MiB operand arena and a 96 MiB link
allocation. The full FP32 word table needs 44.7 MiB. The complete model uses
`model_runtime/board_model.ld` and exports tensors for scheduler readback.

FP32 ordered-add and max reductions use scalar lowering as a software
workaround for the FPGA path. This does not establish a hardware root cause.
The XLNet example documents a remaining 1 ULP difference in its
[README](../xlnet-base-cased/README.md); bitwise comparison treats such a
difference as failure.

Generated weights, images, downloaded BINs, logs and caches stay in ignored
`assets/` and `build/`.
