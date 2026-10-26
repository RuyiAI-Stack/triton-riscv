# Capturing and assembling the complete model

The model sources were ported from branch `vit-base-patch16-224`, commit
`478766cfddb11e61560c44d68320ee21279b90ae` of `CBalaa/buddy-mlir-for-fpga`.
The original 34 operator sources and six added validation/model entries live in their individual case directories. Shared
metadata is in `../cases.py`; compiler orchestration is in `../tools/build.py`.
The original monolithic `triton/kernels.py` is not used.

Run from the repository root with a Python environment that contains PyTorch,
Transformers, SafeTensors and the matching built Buddy bindings:

```bash
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
export PYTHONPATH="$PWD/build/python_packages${PYTHONPATH:+:$PYTHONPATH}"
python "$VIT/tools/fetch_assets.py" --output "$VIT/assets"
python "$VIT/tools/import_model.py" --assets "$VIT/assets" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --output "$VIT/build/import" --save-mlir
python "$VIT/tools/build.py" --all --stage boscame
python "$VIT/tools/simcheck.py" --all
python "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels" --output "$VIT/build/replacement-w8a8" \
  --require-complete
python "$VIT/tools/lower_model.py" --replacement "$VIT/build/replacement-w8a8" \
  --output "$VIT/build/lowered"
python "$VIT/tools/model_simcheck.py" --mode w8a8 \
  --replacement "$VIT/build/replacement-w8a8"
```

W8A8 dense linears are the default on replacement and the model gate. Use
`--fp32-reference` and `--mode fp32` explicitly for the dense floating-point
reference, with a separate replacement output directory (`build/replacement-fp32` by convention). Its composed patch
projection remains quantized; it is not the pure FP32 HuggingFace baseline. The pinned checkpoint
is about 346 MB and is not checked in. `fetch_assets.py` verifies its SHA256
and reuses an already verified copy. Model import requires the real checkpoint;
random initialized weights are not a substitute.

Each numerical kernel follows `TTIR -> standard bufferized contract ->
BOSCAME plan -> plan verify -> emit`. The graph lowerer handles the remaining
standard graph plumbing around those external calls; it neither selects
kernel routes nor compiles numerical operations through a host fallback.

`model_simcheck.py` is currently a prerequisite audit. It checks required case
results and emitted-module hashes. It returns `SKIP` (exit status 2) for any
blocked/missing kernel, incomplete launch coverage or stale evidence. Even if
all kernels pass, it cannot claim full-model simulation until a simulator
executes the complete graph with those emitted kernels and the pinned model
assets. Host graph check scripts remain available only as optional host
checks; no NR board execution is part of this port.


## Reproducible accuracy comparison

Provide a JSON sample manifest with a `samples` list. Each item has a unique
`id`, an image `path` relative to the manifest, and an optional zero-based
ImageNet `label` in `[0,999]`. Missing labels are allowed; top-1/top-5 accuracy
is then reported only for the labeled subset. Both execution paths consume
the exact prepared NPZ bytes and recorded sample order.

```bash
python "$VIT/tools/compare_accuracy.py" prepare --assets "$VIT/assets" \
  --samples samples.json --output "$VIT/build/accuracy-inputs.npz"
python "$VIT/tools/compare_accuracy.py" reference --assets "$VIT/assets" \
  --inputs "$VIT/build/accuracy-inputs.npz" \
  --output "$VIT/build/fp32-logits.npz"
# Run the complete emitted model on accuracy-inputs.npz, producing w8a8-logits.npz.
python "$VIT/tools/compare_accuracy.py" compare \
  --inputs "$VIT/build/accuracy-inputs.npz" \
  --reference "$VIT/build/fp32-logits.npz" \
  --candidate "$VIT/build/w8a8-logits.npz" \
  --report "$VIT/build/accuracy-comparison.json"
```

The reference command verifies checkpoint/config hashes and runs pure FP32
patch/dense operations. The prepare command verifies the pinned processor,
records original image hashes and saves the processor's deterministic output.
Candidate NPZ must contain finite `[samples,1000]` `logits`. Its `.json` sidecar
must copy `model_id`, `revision`, `checkpoint_sha256`, `inputs_sha256`,
`preprocessor_sha256` and `sample_ids` from the prepared-input sidecar, provide
`logits_sha256`, `module_sha256` and `execution_report_sha256`, and explicitly
record `execution="boscame-emitted-simulator"` or `"boscame-hardware"`,
`full_model_executed=true`, and `all_numerical_kernels_emitted=true` from the
actual runner. Host graph checks cannot satisfy this protocol. The complete
emitted-model runner remains unavailable while required operators are blocked.

The comparison reports max/mean/RMS logit error, prediction agreement and
labeled top-1/top-5 changes in percentage points. `MEASURED` records metrics;
it does not imply acceptable accuracy. No dataset or checkpoint is included,
and no full-model accuracy result has been established by these scripts.
