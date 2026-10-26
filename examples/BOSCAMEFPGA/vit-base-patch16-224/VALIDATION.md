# Port validation, 2026-10-26

Upstream source: `CBalaa/buddy-mlir-for-fpga`, branch
`vit-base-patch16-224`, commit `478766cfddb11e61560c44d68320ee21279b90ae`.
Compiler baseline: `cd164388` plus the rectangular-clamp specialization,
segmented boundary-transfer, QK storage-normalization and internal-copy fixes
in this change. No FPGA run or hardware performance measurement is made.

## Current 40-case build

The fresh `build/build-report.json` records **40/40 PASS through restricted NR
assembly and RISC-V object generation**. All 418 `kernel.*` artifact SHA256
values from the previously supported 38 cases match the pre-change snapshot;
there are no artifact differences for those cases. The two nonlinear cases'
emitted modules and objects retain the hashes used by their numerical checks.
The [nonlinear checkpoint](../../../docs/validation/nonlinear-operators/)
records the full build and unchanged-artifact comparison.

The historical complete-inventory checkpoint built 37 of these 40 entries.
The subsequent single-case
[PV K-tail checkpoint](../../../docs/validation/pv-k-tail/) builds the 38th
entry and passes three complete-grid strict-simulator rounds. The subsequent
[nonlinear checkpoint](../../../docs/validation/nonlinear-operators/) builds
GELU and Softmax through restricted assembly and object generation and executes
both complete grids three times. These two entries remove the remaining
inventory compilation blockers; older checkpoint counts below remain scoped
to their recorded builds.

All six quantizers, all four dequantizers, all ten accumulating/overwrite i8
matmuls, all five FP32 linears, LayerNorm, QK, PV, GELU, Softmax, bias/add and the im2col entry
now compile. The historical 18-blocker table below no longer describes the
current compiler.

Compilation is separate from execution evidence. Full-grid FP32 reference
simulations and the complete patch-projection composition have not been run;
their reports identify a validation gap rather than a compilation failure.
Small ordered FP32 regressions cover column-major and general-stride storage.
The emitted W8A8 Linear-chain results and full-model accuracy status are
recorded in the W8A8 section below.

## GELU and Softmax checkpoint

The original erf-form GELU and stable Softmax algorithms are retained. The
example exporter adds a neutral, per-function `contract.math` authorization:
erf absolute error `2e-6`; exp error `8*2^-149 + 2e-6*|exp(x)|`, over all f32
inputs with IEEE value behavior. Other math operations gain no implicit
authorization. Backend plans select versioned piecewise-erf and ln(2)-reduced
exp polynomial recipes; the restricted target uses scalar f32 evaluation and
single-lane vector loads/stores. It adds no host math call or new FMA. Max
reduction uses the explicit `-Inf` initializer and padding, ignores a single
NaN when another argument is numeric, and retains the ordered sum/division.

This is an authorized numerical approximation. Both the original ordered-f32
oracles and independent binary64 mathematical references keep the existing
`rtol=atol=2e-5`. The latter use erfc for GELU's negative tail and math.fsum for
Softmax's mathematical denominator. The measured errors below are finite
output errors against these independent mathematical references:

| Case | Programs/round | Outputs/round | Three-round programs/outputs | Normal max abs error | Stress max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| gelu_197x3072 | 4728 | 605184 | 14184 / 1815552 | 4.25908e-7 | 4.43347e-7 |
| softmax_12x197x197 | 2364 | 465708 | 7092 / 1397124 | 1.60729e-7 | 2.76858e-8 |

Every round covers the complete grid exactly once. Each of 48 contiguous
program shards retains one strict Machine and poisoned SPM through normal
forward, stress reverse and repeated stress forward execution. No global
inter-shard order is asserted. DDR/SPM/REG checks, guards, readonly X, exact
per-program read/write coverage and unchanged unselected output bytes remain
enabled. The two stress rounds have identical output hashes. Per round, GELU
reads and writes 2420736 B each; Softmax reads and writes 1862832 B each.

Stress inputs include erf joins and neighboring f32 values, the GELU negative
tail, NaN, signed zero, infinities, subnormals and maximal finite values.
Softmax covers full 197-lane rows with 59 masked lanes, all-equal rows, offsets,
large dynamic range, exp range-reduction/underflow boundaries and exceptional
rows. GELU(-Inf) follows the original expression and produces NaN. A Softmax
row containing NaN or +Inf, or consisting entirely of -Inf, produces all NaNs;
finite maxima with -Inf elements produce zero probabilities at those elements.

An additional independent 80-decimal-digit mpmath check samples 6482 GELU and
9456 Softmax outputs per round from the actual stored output archives. Maximum
binary64-reference differences from these samples are 1.77636e-15 and
5.55112e-17 respectively. These are sampled observations. Reproducible analytic
primitive and composition bounds are recorded separately in the
[coefficient/error certificate](../../../docs/validation/nonlinear-math/).
No correctly-rounded, one-ULP, full-model accuracy or hardware-performance
claim follows from either check. SFU exp2 is evaluated in that record and is
not selected by this implementation.

The [execution checkpoint](../../../docs/validation/nonlinear-operators/)
retains full per-shard execution metadata and actual-output archive hashes.
`nonlinear-outputs.npz` remains in each build directory. Collection requires
the current source/module hashes and the actual archive, rechecks values
against both oracles, and rejects stale, incomplete or reordered evidence.

## PV K-tail checkpoint

The unchanged PV kernel now lowers K=197/BK=64 as a three-iteration full-K
loop and a fixed five-element tail in both M branches. The declared 16x64
and 64x16 arithmetic shapes, positive-zero padding, zero-initialized product
and separate block-add order are preserved. A's 788 B DDR row pitch uses
existing raw ACC32 copies with valid widths 256/20 B; B and C retain legal
64 B DMA packets. There is no floating AME MMA or new FMA. Planned bank
high-water marks are 10240/9216 B.

Strict emitted-IR simulation passes **1872 programs and 453888 bit-exact f32
outputs** across three complete grids, with maximum absolute error 0. The
48 fixed `(head,N tile)` validation shards retain dirty SPM across their
13 M programs and all three rounds. The normal round is forward in M;
both stress tail-first and reverse rounds use M=12..0 inside each shard,
so the third is a repeated stress/dirty-SPM check. No global inter-shard
order is claimed. The parent requires exactly-once coverage of all 624
programs/151296 outputs per round and reconstructs the result only from
actual executed slices. A/B, DDR guards, logical rows and unselected output
tiles are checked. Each round moves exactly 15923904 external bytes.

The independent PV oracle now models the original floating frontend's
zero-product plus separate add per K block; the previous whole-K sequential
reference was incorrect for this previously failing multi-block case.
The QK reference is unchanged. See the
[PV evidence](../../../docs/validation/pv-k-tail/) for artifact hashes,
actual copy geometries, output hashes, production object audit and the
run-start/current simulator source provenance. These results simulate
emitted IR; assembly execution, FPGA testing and full-model accuracy remain
outside this checkpoint.

## Historical 34-case checkpoint (superseded)

Before the W8A8/zero-accumulator update, an audit exported all 34 cases from
their individual Triton sources and
attempted the frontend/contract verifier/planner/plan verifier/emitter pipeline.
16 completed LLVM lowering, strict NR assembly processing and RISC-V object
generation; the remaining 18 stopped in the frontend. Their blocked-case set
and first diagnostics matched an independent fresh frontend audit.

`attention_qk_12x197x197x64` now compiles through target object generation.
Its complete-grid simulation passes: 2028/2028 programs in one reverse-order
round, with 465708 f32 outputs matching the ordered reference bit for bit and
maximum absolute error 0. Readonly A/B, DDR guards and persistent poisoned SPM
checks pass. The
[QK storage/internal-copy evidence](../../../docs/validation/qk-storage-internal-copy/)
records the build, four-branch comparisons and
[full simulation report](../../../docs/validation/qk-storage-internal-copy/qk-full-simcheck-report.json).

The table combines the 15 cases' complete-grid functional simulation evidence
from the previous checkpoint with the newly completed QK run; it does not
claim that all 16 simulations were rerun in this audit.

| Case | Programs per grid | Simulation rounds | Output comparisons | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| add_197x768 | 1182 | 1 | 151296 | 0 |
| add_1x1000 | 8 | 1 | 1000 | 0 |
| add_1x768 | 6 | 1 | 768 | 0 |
| attention_qk_12x197x197x64 | 2028 | 1, reverse order | 465708 | 0, bitwise ordered f32 reference |
| bias_add_196x768 | 196 | 1 | 150528 | 0 |
| bias_add_197x3072 | 591 | 1 | 605184 | 0 |
| bias_add_197x768 | 197 | 1 | 151296 | 0 |
| bias_add_1x1000 | 1 | 1 | 1000 | 0 |
| bias_add_1x768 | 1 | 1 | 768 | 0 |
| layernorm_197x768 | 197 | 1 | 151296 | 0, ordered f32 reference |
| layernorm_1x768 | 1 | 1 | 768 | 0, ordered f32 reference |
| matmul_i8_196x768x768 | 84 | 3 | 451584 | 0 |
| matmul_i8_197x3072x768 | 336 | 3 | 1815552 | 0 |
| matmul_i8_197x768x3072 | 84 | 3 | 453888 | 0 |
| matmul_i8_197x768x768 | 84 | 3 | 453888 | 0 |
| matmul_i8_1x1000x768 | 16 | 3 | 3000 | 0 |

Integer outputs match `int64` NumPy dot plus existing C, cast to i32 to model
wrap. Stress rounds use signed-i8 extremes and C values near both i32 limits.
Two stress runs use the same input with tail-first and reverse program order
and require identical output hashes. Pointwise arithmetic is exact f32.
LayerNorm acceptance is `rtol=atol=2e-5` against an independent ordered-f32
reference; the recorded generated outputs happen to match it exactly.

The previous 15-case checkpoint executes 4192 programs and compares 4391816
output elements. Adding the new QK run gives 6220 programs and 4857524 output
comparisons. Counts include all three complete grids for each i8 case, rather
than counting a stress case as a single launch.

The interpreter executes the actual `kernel.boscame.mlir`, with only standard
Linalg-to-loop/affine cleanup for interpretation. DDR bounds, guards, readonly
inputs, DMA rules and distinct DDR/SPM/register access kinds remain checked.
SPM is poisoned and persists through every program of a launch.

## Historical blockers (superseded)

At the previous checkpoint, the remaining 18 cases retained their source and
independent launcher. They were
reported as frontend `BLOCKED`, not successful simulations. These are the
first diagnostics from that historical audit, grouped into seven compilation
behaviors below. There are five distinct first-diagnostic strings: GELU and
Softmax share V25, while Patch Embed and quantization share the scalar-memory
diagnostic.

| Cases | First stage / diagnostic | Why this is left for design |
| --- | --- | --- |
| attention_pv_12x197x64x197 | Frontend V4: static valid shape must be positive and match access rank | Its K=197, BK=64 loop has a final block with 5 valid K elements. Internal loop specialization must preserve zero padding and floating-point accumulation order; current program full/tail specialization does not cover it. |
| all 5 linear_* cases | Frontend V19: cross-scope edge is not a registered loop init/exit edge | Bufferization preserves a zero tensor read both outside and inside the K loop. This is invariant capture, not a loop-carried slot; V1 has no invariant-capture schema. |
| all 4 dequantize_* cases | Frontend: neutral-contract gather/scatter load unsupported in V1 | Row/column scale lookup uses index/COLS and index%COLS. Define masked indexed loads with repeated-index semantics, or prove an equivalent scalar-broadcast/contiguous-segment normalization. |
| patch_embed_196x768 | Frontend: unstructured/scalar memory access unsupported in V1 | im2col uses scalar gather; its full projection also depends on blocked quantize/dequantize cases. |
| all 5 quantize_* cases | Frontend: unstructured/scalar memory access unsupported in V1 | The scale output is a scalar DDR store. Subsequent recipes must preserve abs/max reduction, zero-row scale=1, ties-away-from-zero rounding, saturation and f32-to-i32-to-i8 casts. |
| gelu_197x3072 | Frontend V25: generic requires one pure arithmetic operation (`math.erf`) | Exact erf needs an explicit mathematical/target implementation; a different approximation changes numerics. |
| softmax_12x197x197 | Frontend V25: generic requires one pure arithmetic operation (`math.exp`) | Exp and max reduction need target recipes, including NaN/Inf and invalid-lane -Inf semantics, with the original sum/division order preserved. |

The five ordinary FP32 linears are `linear_196x768x768`,
`linear_197x3072x768`, `linear_197x768x3072`,
`linear_197x768x768` and `linear_1x1000x768`. The four dequantizers
are `dequantize_196x768`, `dequantize_197x3072`,
`dequantize_197x768` and `dequantize_1x1000`. The five quantizers
are `quantize_196x768`, `quantize_197x3072`, `quantize_197x768`,
`quantize_1x768` and `quantize_768x768`.

These are first blockers, not a guarantee that resolving each one alone
completes the case. Max reduction and new math operations still need
verified target recipes; external boundary-copy support does not add those
computational semantics. FP32 matmul already has a scalar, ordered-mnk
route with preserved floating-point policy, used by the now-compiling QK.
At that checkpoint the remaining linear restriction was loop invariant
capture; the current FP32 linears compile through object generation.

For V19, the zero initializer is copied into the loop-carried accumulator,
and independently copied into a temporary product on each iteration.
Relabeling that second edge as a loop init would be incorrect. The verifier
has therefore not been weakened, and no incomplete graph is silently planned.

At that checkpoint, those design-dependent cases and both full-model
verification modes were skipped. `model_simcheck.py` still returns `SKIP`
and exit status 2; no model inference/top-1 accuracy claim is made. Model
weights, host graph execution and a board image are not used to disguise this
gap.

## Direct fixes

### Exact boundary transfers

The three previously blocked 1000-column cases now build through RISC-V
object generation and pass the strict emitted-IR simulator:

| Case | Actual planned route | Verification |
| --- | --- | --- |
| add_1x1000 | Full input tiles use 512 B DMA and terminal outputs use existing direct RVV forwarding. The last 104 f32 input/output views use 384 B DMA plus 32 B RVV. | 8/8 programs, 1000 outputs, exact f32 match. |
| bias_add_1x1000 | Bias uses 3968 B DMA plus 32 B RVV. X and output use exact RVV (3968 B in 64 B steps plus a 32 B tail), because the dynamic row-offset residue is not proven. | 1/1 program, 1000 outputs, exact f32 match. |
| matmul_i8_1x1000x768 | Full C rows use 256 B DMA despite unused 4000 B parent pitch. The final N=40 C view uses 128 B DMA plus 32 B RVV in both directions. B retains physical [1000,768] storage and valid physical-row DMA; padded AME computation is unchanged. | 3 complete 16-program grids, 3000 outputs, exact i32 match including nonzero C, signed extremes, wrap and tail-first/reverse order. |

The generated NR assembly contains `vsetivli zero, 8, e32, m2` for the
32 B tails (three sites for each add case, two for matmul), with ordinary
element loads/stores. No whole-register vector access or CSR instruction
is used. Guards, readonly inputs, poisoned SPM and DMA legality remain
checked by the existing simulator. No external padding is added.

The new regression covers same/different endpoint residues, exact RVV
prefix/body/tail, both directions, short copies, multirow address drift,
single-row unused pitch, completion after the final segment, malformed
plans and padded matmul N tails. The new segmented recipe currently
rejects 16-bit RVV copies because the NR target does not permit the needed
`vle16/vse16` forms; wholly legal 16-bit DMA is retained. This is a target
implementation limit, not a DMA requirement on all AME/RVV operations.

### Frontend and launcher fixes

The frontend previously recognized only
`min(max(min(start+tile,bound),start)-start,tile)`. Canonicalization can remove
the redundant outer min. The matcher now accepts both forms while preserving
the same signed/no-wrap launch and loop-domain proof. Launch-proven scalar
selects such as `row < ROWS ? valid_columns : 0` are also folded. Unknown
predicates and unproven domains remain rejected. This restores the two add
and four bias-add cases without changing numerical operations or DMA rules.

The imported C launchers also had physical-layout bugs: transposed logical B
descriptors instead of physical `[N,K]`, unaligned allocations, and overlapping
row/column dequantization scale storage. Only their descriptors/offsets were
corrected. Native ABI smoke checks cover all 34 launchers and 25808 raw program
entries, including patch composition's four child calls. They explicitly
report `numeric_oracle_executed=false` and `simulator_executed=false`.
A misaligned-buffer negative check traps before raw kernel entry.

Graph-call adapters explicitly construct typed descriptors when removing a
leading unit batch dimension: LayerNorm rank 3 becomes rank 2 and attention
rank 4 becomes rank 3. They preserve nonzero element offsets, validate shape
and contiguous strides, and align static output storage to 64 bytes. W8A8
accumulator address/nonalias checks run before output initialization. The
replacement index admits only successful, current builds with matching source,
metadata, TTIR, adapter and emitted-module hashes. The native
`vit-model-adapters.py` regression covers valid and invalid descriptors and
rejects blocked, TTIR-only and stale replacement candidates. These tests
validate graph ABI construction and indexing, not whole-model execution.
Independent case builds and simulations write their summaries inside the case
directory; subset reports also have a separate path. They preserve the suite
reports used by the model prerequisite audit.

## LayerNorm semantic correction

The original kernel reduced squared padded lanes after subtracting the mean.
For WIDTH=768 and BLOCK=1024, those 256 invalid lanes became `-mean` and inflated
variance. Both kernels now use `where(valid, x-mean, 0)` before squaring.
This is a semantic bug fix and **is not bit-identical to the defective port**.

On the same deterministic seed-71 197x768 inputs, both old and corrected emitted
IR were executed for all 197 programs with the strict interpreter:

| Reference | Original maximum absolute error | Corrected maximum absolute error | Corrected elements outside 2e-5 tolerances |
| --- | ---: | ---: | ---: |
| Standard valid-width LayerNorm, float64 | 0.0654971982 | 0.00000252608 | 0 |
| PyTorch float32 `layer_norm` | 0.0654973984 | 0.00000286102 | 0 |

The corrected IR exactly matches the ordered-f32 oracle on this input. This
does not assert bitwise equivalence to PyTorch's different reduction order or
establish whole-model accuracy. `tools/check_layernorm_numerics.py` reproduces
the current check; `--baseline-module` optionally compares saved pre-fix IR.

## Evidence and reproduction

The checked-in
[QK storage/internal-copy evidence](../../../docs/validation/qk-storage-internal-copy/)
records the additional QK build and branch checks, internal-copy guarantees,
artifact hashes and the completed
[QK full-grid simulation](../../../docs/validation/qk-storage-internal-copy/qk-full-simcheck-report.json).
The earlier 15-case simulation checkpoint remains identified separately from
the new QK result and their combined totals.

Generated evidence is intentionally ignored by Git:

- `build/build-report.json` and each `build/kernels/<case>/build-result.json`:
  actual completed stage, failures and artifact SHA256 values.
- `build/simcheck-report.json` and per-case `simcheck-result.json`: complete
  grids, references, stress rounds, outputs, errors and module/source hashes.
- `build/launch-check/report.json`: ABI-only checks of the current launch and
  generated adapter sources.
- `build/layernorm-numerics.json`: pre-fix/corrected/reference numerical errors.
- `build/model-simcheck-*.json`: explicit skipped model checks and prerequisites.
- `build/report-isolation.json`: actual single-case build/simulation collection
  and a two-case collection preserve both complete-suite report hashes.

Run the commands in [README.md](README.md) to regenerate them. `--collect`
reuses a simulation only after source/metadata/module hashes, full grid counts
and stress-round coverage have been validated; stale or edited evidence is
rejected. Global suite reports count all 34 cases rather than only the subset
that passed.

## Repository checks

The final `ninja -C build check-buddy -j2` run passes: 239 example tests pass
with 2 unsupported; 619 regression tests pass with 2 unsupported and 1 expected
failure. There are no unexpected failures. The regression set includes the
canonical-clamp positive/negative specialization checks, native model
descriptor/index checks, QK storage normalization, zero accumulators,
quantized pointwise expressions, restricted merge expansion and ordered
FP32 storage layouts.

Task-scoped pre-commit checks pass for the new example, compiler fix and both
regressions. `git diff --check` and `git diff --cached --check` also pass.
No unrelated files are formatted, and no files are staged or committed.


## W8A8 and zero-initialized model products

The model replacement and prerequisite gate now default to W8A8 dense
linears. `--fp32-reference` / `--mode fp32` retain the explicit floating-point
dense path; its patch projection remains the original quantized composition.
The original 34-case checkpoint above precedes this change. Five dedicated
`matmul_i8_overwrite_*` entries and `quantize_1000x768` bring the inventory to
40; they do not replace the independent nonzero-C `matmul_i8_*` tests.

The overwrite source explicitly contains a zero tensor initial value and
never loads external C. The actual neutral classifier entry has local i32
`fill(0)` before its six K iterations and only a terminal external C store.
This lets planner/emitter use the dedicated Acc clear operation at reduction
start. It does not authorize clearing the accumulator on every K tile, moving
Acc lifetime across DMA, or removing unrelated A/B/C tail padding fills.
Graph adapters and the composed patch adapter no longer initialize DDR C.
The compiled-kernel index checks overwrite metadata and the current generated
adapter, so an old accumulating child entry cannot satisfy the new graph ABI.

The 73-dense graph structure check passes with 219 explicit mixed-dtype calls
and rejects falling back to an accumulating matmul when overwrite is missing.
The native adapter regression checks output offsets, unchanged incoming C at
the call boundary and preservation of the old accumulating ABI. These checks
validate structure/ABI, not whole-model numerics.

Quantization retains per-row/per-channel symmetric [-127,127] scales, zero-row
scale 1 and nearest ties away from zero. Its old f32 `ratio +/- 0.5` intermediate
could round `nextafter(0.5,0)` to 1 before integer conversion. The source and
independent launch/oracle/weight-packing paths now use truncation plus a
fractional comparison. This corrects that boundary result from 1 to 0 and
therefore is a numerical change from the defective quantizer; exact half ties,
saturation endpoints and zero rows retain the intended rules. Dequantization
uses proven per-row scalar scale broadcasts and contiguous column-scale
loads, preserving `(float32(acc)*row_scale)*column_scale` order.

`tools/check_w8a8_chain.py` is the emitted-chain acceptance entry: quantize
activation/weights, overwrite AME dot, dequantize, bias and residual. It records
actual module hashes, guards/readonly accesses, polluted persistent SPM,
repeated workspace reuse, zero rows, half neighbors and different scales.
Its final `build/w8a8-chain-check.json` is **PASS** for four complete stress
rounds (4136 emitted programs). Quantization, integer accumulation and ordered
dequantization/bias/residual match independent oracles exactly. The first
forward round and the reverse round reusing previous accumulator contents
produce the same output hash, `6fd45bf8e66f0ece5dba6b00939e45e8cf66098c4ae9bc5431496f130d64ec22`.
The changed-token-scale and zero-token rounds also pass. No old external C
read is observed. Module/source hashes are checked against the frozen build,
and cached simulator Machines are keyed by module hash.

The historical W8A8 full-inventory `build/simcheck-report.json` contains 31 PASS cases,
17485 executed programs and 11211403 output comparisons. Unchanged earlier
evidence is reused only after complete-grid/round/source/module validation;
in particular QK's current source and emitted module exactly match its
archived 2028-program simulation. The nine non-PASS entries comprise the
three compilation blockers at that checkpoint and six explicitly unexecuted
validations (five FP32 reference full grids and the composed patch pipeline).
The FP32/patch entries compile; they are not compiler-blocked operators.
The newer PV checkpoint above adds complete-grid evidence for PV while
retaining this historical report unchanged.

| Current emitted checks | Cases | Programs | Output comparisons | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: |
| Per-row/channel quantize | 6 | 2359 | 2267959 | 0 |
| Ordered dequantize | 4 | 7094 | 908008 | 0 |
| Model overwrite i8 matmul, three rounds each | 5 | 1812 | 3177912 | 0 |
| Preserved nonzero-C i8 matmul, three rounds each | 5 | 1812 | 3177912 | 0 |

All 40 standalone descriptor/grid adapters pass the native ABI smoke check;
the separate 73-dense rewrite check passes with 219 external calls. Neither
result is a complete-model execution. Final production assembly retains the
required adjacent `fence rw,rw` instructions. Quantizer `vmerge.vim` selections
are expanded into the already supported `vmerge.vxm` form with a saved scalar
scratch register and 16 bytes of temporary scalar stack. The allowed ISA set
is unchanged. All 37 successfully generated objects, including all six
quantizers and five overwrite entries, pass the production ELF checker.
LayerNorm uses the existing exact `vfsub.vf` expansion.

The chain's optional ordered-FP32 dense comparison reports quantization error
on synthetic boundary inputs (maximum absolute error 1052592 at the largest
synthetic channel scale). It is not a checkpoint-logit or model-accuracy
measurement and is not an accuracy acceptance result.

`tools/compare_accuracy.py self-test` validates metric computation only. The
accuracy workflow pins checkpoint/preprocessing/sample hashes, prepares one
shared input NPZ, computes an explicit pure-FP32 eager reference, and accepts
candidate logits only with complete emitted-model execution evidence. Missing
checkpoint, validation images or candidate execution results produce SKIP.
No complete-model logits, prediction agreement or labeled top-1/top-5 result
has been measured here, and no accuracy acceptance conclusion is asserted.
