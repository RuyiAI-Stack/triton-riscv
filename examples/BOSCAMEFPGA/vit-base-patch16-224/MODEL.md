# ViT graph contract

The imported model is the complete pinned `google/vit-base-patch16-224`
checkpoint: batch 1, image `[1,3,224,224]`, 196 image patches plus one CLS
position, 12 encoder blocks, hidden width 768, 12 heads of width 64, MLP width
3072, and 1000 output logits. `model.json` records the immutable model revision
and checkpoint digest.

`tools/import_model.py` keeps the HuggingFace ordering and all 200 captured
parameters. It spells attention as QK matmul, scale `1/sqrt(64) = 0.125`,
softmax, and PV matmul. Dense biases remain separate operations. Exact GELU
uses `erf`; LayerNorm uses epsilon `1e-12`. Input pixels are already normalized
FP32 values; resize/crop and mean/std `(0.5,0.5,0.5)` are outside the graph.

`tools/replace_triton.py` matches shapes and dataflow and emits external calls.
Its kernel index accepts only successful current contract-pipeline builds;
an exported manifest from a blocked lowering cannot satisfy graph coverage.
Graph descriptors with a leading unit batch dimension are checked and rebuilt
at the operator's rank while preserving their element offset. Descriptor
structures of different ranks are never cast, and result storage is aligned
to the caller contract's 64-byte boundary.
The ordinary graph has 245 numerical calls: one patch projection, 73 dense
projections, 73 biases, 25 LayerNorm calls, 12 each QK/softmax/PV/GELU calls,
and 25 additions (24 residuals and position embedding). Reshape, transpose,
CLS concatenation and indexing remain graph plumbing. The four linear shapes
are `[M,N,K] = [197,768,768], [197,3072,768], [197,768,3072], [1,1000,768]`.
Physical checkpoint weights are `[N,K]`; their transpose is internal to the
linear kernel and must not be applied twice.

W8A8 is the default AME model path (`--w8a8` remains explicit). Each of the
73 linears becomes activation quantization, signed
INT8 matmul and dequantization; biases remain explicit. Quantization is
per-token/per-output-channel, symmetric `[-127,127]`, nearest ties away from
zero, with zero-row scale 1. Rounding truncates the clamped ratio, compares its
fraction with +/-0.5, and adjusts the integer. This avoids an intermediate f32
`+0.5` rounding a value immediately below a half integer into the next integer.
The replacement records the exact mixed-dtype
parameter and workspace ABI; shape alone cannot identify equal-sized tensors.

Model calls use dedicated `matmul_i8_overwrite_*` entries with `C=A*B` semantics.
Their real Triton source starts from a zero tensor and stores the product without
loading external C. Standard neutral IR therefore contains a local `fill(0)`
initial accumulator and the full K-loop dependency. The planner may replace
proven zero initialization by `bosc_ame.tile.zero_acc` at the reduction start;
later K iterations still reload the stored accumulator, without extending Acc
register lifetime across DMA/runtime calls. Padding fills remain independent.
No graph adapter clears DDR C. Existing `matmul_i8_*` entries keep their
`C+=A*B` contract, nonzero C input and integer-wrap tests.

`--fp32-reference` explicitly selects floating-point dense kernels. It does
not make the existing composed patch projection FP32. Biases and residuals
remain float operations after `(int32_product * row_scale) * channel_scale`;
the two dequantization multiplies are never reassociated. Each row/channel
scale applies to the entire K reduction, not independently to K tiles.

The upstream patch projection is itself composed: the Triton patch kernel
only writes `[196,768]` im2col. Its external graph adapter quantizes patches
and flattened `[768,768]` weights, executes the signed INT8 matmul,
dequantizes, applies bias, and returns NCHW `[1,768,14,14]`. An ordinary wrapper
around the im2col kernel is not a complete patch embedding implementation.
Thus the ordinary linear mode is mixed precision for patch projection; it
must not be described as a bit-exact FP32 implementation of HuggingFace.

There are 40 independently launchable shape specializations: the original
34, five model overwrite matmuls, and a 1000-row weight-quantization check, including
additional diagnostics and non-model shapes. The ordinary model requires 18
of them, counting the four composed-patch dependencies; the W8A8 model
requires 24. `tools/model_simcheck.py` records those requirements explicitly.
Kernel simulation uses the actual contract/planner/emitter output. A host
ExecutionEngine check is optional numerical evidence and is never a model
simulation PASS. If a compiler limitation blocks a required operator, the
model report must state `SKIP` and the concrete limitation.


## Emitted chain and model accuracy

`tools/check_w8a8_rewrite.py` checks the 73 dense graph substitutions and the
mixed-dtype descriptor ABI without claiming numerical validation.
`tools/check_w8a8_chain.py` runs activation and weight quantization, overwrite
AME matmul, dequantization, bias and residual through current emitted kernels.
It checks independent integer/f32 oracles, different old workspace contents,
repeated calls with persistent polluted SPM, reverse/tail-first order, zero rows,
half-integer neighbors and token/channel scales. All four stress rounds pass
the integer and ordered quantized-FP32 oracles exactly. A chain PASS establishes
this Linear chain; it does not establish complete-model execution or accuracy.

The complete-inventory checkpoint built 37 of 40 entries through restricted
target assembly; the subsequent PV K-tail checkpoint confirms the 38th entry
and three complete-grid numerical rounds. The later nonlinear checkpoint
builds GELU and Softmax and passes both complete grids in three strict-simulator
rounds under explicit approximation contracts and the existing `2e-5` output
tolerances. A fresh complete-inventory assembly/object build passes all 40
entries, with all 418 kernel artifacts from the prior 38 cases unchanged.
This does not establish complete-model execution or accuracy. The FP32 reference entries and patch im2col compile,
but their large full-grid/reference-composition execution has not been verified.

The explicit FP32 accuracy reference in `tools/compare_accuracy.py reference`
uses the pinned checkpoint and the eager import wrapper, including a floating-
point patch projection. This is different from `--fp32-reference` in the graph
replacement, whose patch projection remains the existing quantized composition.
See [model/README.md](model/README.md) for the sample/logit exchange protocol.
Missing checkpoint, samples or complete emitted-model output is reported as
`SKIP`. Metric computation has no default accuracy acceptance threshold.
