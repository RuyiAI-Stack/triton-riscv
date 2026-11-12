# XLNet-Base-Cased on BOSCAME

This example follows `../vit-base-patch16-224` for its launchers, Triton
operators, typed adapters, NH/E6 runtime and linker scripts. The pinned
checkpoint is
[`xlnet/xlnet-base-cased`](https://hf-mirror.com/xlnet/xlnet-base-cased).
Each operator directory has `kernel.py`, a literal `KERNEL_CONFIG`,
`int launch(void)` and a `makefile`. `common.mk` and `board.mk` own the
shared compile and bare-metal link.

The fixed specialization is batch 1, sequence 128, 12 layers, hidden 768,
12 heads, head dim 64, intermediate 3072 and vocabulary 32000. Evaluation
uses the bidirectional content stream with `use_mems=False`. Attention masks
and token type ids 0/1/2 are supported. Training dropout, the query stream,
recurrent memory, dynamic shapes and the unidirectional configuration are
not.

The full word embedding stays FP32. Linear layers use per-row symmetric
W8A8, signed i8/i32 matmul and FP32 dequantization. The integer range is
[-127, 127], rounding is nearest ties away from zero, and an all-zero row
uses scale 1. `model_runtime/model.c` runs the full encoder and an optional
LM head. The LM head uses 32 tiles of 1024 words and keeps the last 256,
producing `logits[128,32000]`.

Q/K/V/R weights in `[input,head,dim]` are stored as physical
`[output,input]`; O is already `[N,K]`. QK, position scores, segment scores
and PV run per head. The scale 0.125 is applied once after the AC+BD+EF
merge. Segment N=2 uses a scalar int32 row-dot with BM=BN=1 and BK=64; the
other matmuls use AME. `cases.py` checks that every BM/BN/BK fits the real
M/N/K.

## Prepare the environment and inputs

Run from the repository root. This needs a built LLVM, Buddy and the Triton
CPU backend, plus the Python environment used to prepare model inputs. The
fixed revision and hashes are in
[model/config.json](model/config.json) and
[model/manifest.json](model/manifest.json). `assets/` holds the downloaded
checkpoint and tokenizer. `build/` holds generated files. Both are gitignored.

```sh
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
X=examples/BOSCAMEFPGA/xlnet-base-cased
K="$PWD/$X/build/kernels"

python "$X/tools/fetch_assets.py" --weights
python "$X/tools/prepare_weights.py" --lm-head --output "$X/build/weights-lm"
python "$X/tools/prepare_model_input.py" --lm-head --output "$X/build/model-input-lm"
```

`prepare_weights.py` writes weights and scales in the physical layout and
keeps the FP32 embedding. `prepare_model_input.py` left-pads a real sentence
pair to 128 tokens. Relative positions 128 through -127 are generated on the
host. The content-stream attention mask does not mask the diagonal, including
padding query rows. `model.c` turns `mask[128]` into `non_target[128,128]`.

## Compile operators and the complete model

```sh
make -C "$X" launchcheck
make -C "$X" model-runtime-object
make -C "$X" -j4 asm KERNELS="$K"
make -C "$X" -j4 bin KERNELS="$K"
make -C "$X" bincheck KERNELS="$K"

python "$X/tools/build_model_bin.py" --lm-head --kernels "$K" \
  --weights "$X/build/weights-lm" --model-input "$X/build/model-input-lm" \
  --output "$X/build/model-bin-lm"
python "$X/tools/build_model_host.py" --build "$X/build/model-bin-lm" \
  --kernels "$K" --output "$X/build/model-host-lm"
```

`launchcheck` checks the grid and descriptor ABI. `bincheck` checks source,
build-product and ELF-audit hashes, and extracts the BIN from the final ELF
for a byte compare. The complete-model builder links the real `model.c`, AOT
objects, typed adapters, the NH/E6 runtime and freestanding math.
`build_model_host.py` runs the same C graph and emitted LLVM on the host.
AME integer matmul uses an exact signed-i8/i32 dot. The output manifest comes
from the model's `output-layout.json`.

Drop `--lm-head` on the weight and model builds, and optionally on input
preparation, when every encoder hidden snapshot is required. Use a new output
directory. The encoder image exports the embedding and 12 hidden layers. The
LM image exports full-vocabulary logits. `--kernels` and `--output` keep
those builds apart.

FPGA builds default to C++ lowering with `scalar-f32-max-reductions=true`
and the existing ordered-add scalarization. FP32 max uses bounded scratch,
volatile scalar loads and a scalar accumulator. Other vector arithmetic
stays. The final ELF audit rejects a remaining `vfredmax.vs`. The two
policies are independent; changing either rebuilds from the LLVM stage.
This is a software workaround. The hardware cause is not established.

Every entry requires contiguous, non-aliasing, 64-byte-aligned buffers of
the fixed shape. Token ids are in `[0, 32000)`. The complete model uses
`board_model.ld`. NH and E6 entries are `0x80000000` and `0x80010000`.
Parameters start at `0x81000000`. Linker ASSERTs check code, snapshots,
parameters, scratch and the stack. The LM output is 16,384,000 bytes and
needs a scheduler capture window of at least 16 MiB.

## Execute, download and verify

Acceptance is bitwise equality of the local emitted outputs and the BINs
the scheduler actually returns. The board only computes and captures. After
download, the local check compares FP32 bit patterns and raw integer bytes.

Select the FPGA target and confirm the SSH/API forward, token and a READY
device before running. The URL below is the local forward for fpga112.
Another deployment uses its own endpoint.

```sh
export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"

python "$X/tools/run_model_board.py" --build "$X/build/model-bin-lm" \
  --reference-emitted "$X/build/model-host-lm" \
  --output "$X/build/model-runs/run-001" --timeout 7200
```

The client defaults to the sibling `fpga-scheduler` checkout, then to
`fpga-scheduler` on PATH. `--client`, `--endpoint` and `--token-file`
override the connection, as do `FPGA_SCHEDULER_CLIENT`,
`FPGA_SCHEDULER_URL` and `FPGA_SCHEDULER_TOKEN_FILE`. The foreground client
holds the lease until the task ends. Resume and download use the same
configuration. The same `--output` looks up the existing task and checks
the frozen files; it does not submit again.

To check files that are already downloaded, write the new report somewhere
else. This path does not contact the scheduler:

```sh
python "$X/tools/run_model_board.py" --validate-only "$X/build/model-runs/run-001" \
  --validation-output "$X/build/model-reviews/run-001.json"
```

Standalone operator return uses the shared client. Select cases as needed:

```sh
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model xlnet-base-cased --kernels "$K" --case quantize_128x768 \
  --output "$X/build/operator-runs"
python "$X/tools/verify_quantize_bits.py" \
  --runs "$X/build/operator-runs" --case quantize_128x768 \
  --output "$X/build/operator-reviews/quantize_128x768.json"
```

Q and scale snapshots are isolated by a full cache line, so a later call
does not overwrite an earlier output. The downloader checks name, dtype,
shape, nbytes and SHA. A missing, truncated or corrupt file fails the local
check.

## Known difference

Strict encoder comparison still has a 1 ULP difference at
`layernorm_128x768[34,334]` in layer 3. The expected value is
`-1.411880612373352` / `0xbfb4b881`. The actual value is
`-1.4118804931640625` / `0xbfb4b880`. The hardware cause is not established.
The strict compare keeps the difference and does not widen the tolerance.
