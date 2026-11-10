# E5-small-v2 on BOSCAME FPGA

This example adapts the pinned `intfloat/e5-small-v2` checkpoint using the same
per-operator layout, `int launch(void)` entry, typed MemRef ABI, compiler stages
and NH/E6 link flow as `../vit-base-patch16-224`.

The complete graph exports the sentence embedding, final hidden state and 13
hidden-state checkpoints. Use scheduler downloads and local emitted execution
to validate every complete output.

## Model and layout

The fixed specialization uses batch 1, sequence length 128, 12 encoder layers,
hidden width 384, FFN width 1536, and 12 attention heads of width 32. LayerNorm
uses FP32 with epsilon `1e-12`; activations use exact-definition GELU.
Dense layers and per-head QK/PV products follow the ViT W8A8 flow: per-row
symmetric INT8 quantization with ties away from zero, overwrite INT8 matrix
multiplication with INT32 accumulation, then FP32 dequantization. Other
calculations remain FP32. Every matrix tile fits its logical M/N/K dimensions.
The K=32 QK specialization uses exact scalar INT32 dots because AME requires
K>=64; the remaining matrix specializations retain AME operations.

E5 sentence output is the attention-mask-weighted mean of final hidden states,
followed by L2 normalization. Tokenization adds `query: ` or `passage: `;
padding/truncation produces 128 tokens. The model rejects invalid IDs, nonbinary
masks and an all-zero mask.

| Path | Contents |
| --- | --- |
| `<operator>/` | 30 independent kernels, C launchers and makefiles |
| `cases.py`, `common.mk`, `board.mk`, `board*.ld` | Inventory, compilation and linking |
| `model/`, `model_runtime/` | Pinned metadata and complete 1,649-call C graph |
| `tools/` | Asset packing, builds, local execution and scheduler validation |
| `assets/`, `build/` | Ignored checkpoint files and generated outputs |

## Reduction policy

The normal FPGA build enables C++ lowering options
`scalar-f32-ordered-reductions` and `scalar-f32-max-reductions`. FP32 max uses
the compiler's scalar accumulator and volatile scratch accesses; other vector
operations remain enabled. No model kernel, assembly rewrite or binary patch
implements this workaround. The final ELF audit rejects `vfredmax.vs` and the
affected quantizers contain scalar loads and max comparisons.

`ALLOW_VECTOR_MAX_REDUCTIONS=1` opts into vector max for hardware diagnostics.
`ALLOW_VECTOR_ORDERED_REDUCTIONS` independently controls ordered FP32 sum.
The shared policy stamp causes affected stages to rebuild when either option
changes. Non-FPGA compiler defaults stay unchanged. See the shared
[scalar-max guide](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-max-reductions.md).
This is a software workaround; it does not establish the hardware root cause.

## Build and local execution

Run from the repository root in the configured `boscame` environment:

```bash
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
E5=examples/BOSCAMEFPGA/e5-small-v2

make -C "$E5" assets PYTHON=python
make -C "$E5" bin PYTHON=python
make -C "$E5" model-bin PYTHON=python
make -C "$E5" model-host PYTHON=python
```

`model-bin` packs the pinned checkpoint and input, records the static dispatch
schedule, links the C graph with all real kernel objects and audits the final
ELF. It does not generate numerical acceptance references. `model-host`
explicitly executes the same C graph and emitted scalar LLVM locally; matrix
products use exact C INT8/INT32 dots. Its three complete outputs and provenance
manifest default to `build/model-host` (`MODEL_HOST=PATH` overrides it).
`BUDDY_ROOT` and `LLVM_BIN` select the compiler checkout and LLVM tool directory.

## Execute, download and verify

The sole complete-model numerical acceptance criterion is byte equality of
**every complete scheduler-downloaded output against the frozen local emitted
execution**. Shapes, byte lengths, SHA256, task/image identity, successful exit,
DDR readback and reference provenance must also pass. No numerical tolerance is
used. The runner requires an existing emitted reference; it never regenerates
one automatically. Before scheduler access it freezes the outputs, manifest
and source hashes and checks emitted LLVM against the linked kernel objects.

After selecting fpga112 and establishing its scheduler connection:

```bash
python "$E5/tools/run_model_board.py" \
  --build "$E5/build/model-board" \
  --emitted-reference "$E5/build/model-host" \
  --output "$E5/build/model-board-runs/my-new-run" --timeout 1800
```

The foreground client keeps its lease alive. The board computes and captures
raw tensors; the runner downloads the actual BINs and verifies them locally.
It validates the image, successful exit and complete output metadata. On a
connection error it queries the original task and stops without resubmitting.
A resumed run uses the reference frozen before its original submission.

`--client`, `--endpoint` and `--token-file` override the scheduler executable,
fpga112 URL and credential file. Their environment equivalents are
`FPGA_SCHEDULER_CLIENT`, `FPGA_SCHEDULER_URL` and `FPGA_SCHEDULER_TOKEN_FILE`.
Defaults are the sibling scheduler virtualenv (or PATH),
`http://127.0.0.1:18762`, and `.agent/secrets/fpga112.token`. Credentials stay
outside this example.

Recheck downloaded outputs with no scheduler requests and write a **new** review
directory, leaving the original files untouched:

```bash
python "$E5/tools/run_model_board.py" --validate-only \
  --output "$E5/build/model-board-runs/my-new-run" \
  --emitted-reference "$E5/build/model-host" \
  --review-output "$E5/build/review-emitted-new"
```

`tools/verify_model_emitted.py --run RUN --reference REFERENCE --output NEW`
provides the same local check without scheduler code. Missing references,
changed source/object provenance, corrupted downloads, and even one differing
FP32 bit fail closed.

The source checkout contains the tutorial, model configuration and reusable
build/run tools. Checkpoints and generated files stay under the ignored
`assets/` and `build/` directories.
