# FlagGems validation on x86 and QEMU

This workflow validates the FlagGems examples with the Triton-RISCV CPU
backend on an x86 host and checks representative generated RV64 executables
with QEMU.

## x86 test suite

After building Triton-RISCV and sourcing `scripts/triton-riscv-env.sh`, run:

```sh
pytest python/examples/ \
  --ignore=python/examples/test_core.py \
  --ignore=python/examples/test_annotations.py \
  -v
```

The 2026-07-11 A800 validation result was:

```text
3887 passed, 15 skipped in 885.32s (0:14:45)
```

No test was disabled as part of the FlagGems compatibility work. The skipped
cases are guarded by existing environment capability markers, including tests
that require execution on native RISC-V hardware with RVV. Their corresponding
RISC-V object-generation tests still run on x86.

## QEMU smoke validation

Build the RISC-V GNU toolchain and QEMU as described in
[`06-RISCV-QEMU.md`](06-RISCV-QEMU.md), source the environment helper, then run:

```sh
python python/examples/rvv_vec_add_elf.py --dump-asm
PYTHONPATH="$PWD/python/examples" \
  python python/examples/flaggems_qemu_smoke.py
```

The FlagGems smoke program compiles and executes four backend patterns:

- `where`: byte-backed boolean conditions and vector selection;
- `vdot`: scalar reduction without atomics;
- `var`: multi-row scalar reduction with multiple program IDs;
- `upsample_nearest1d`: output-centric index mapping.

Each case generates a static RV64 ELF, runs it with `qemu-riscv64`, and checks
the output in the generated C runner. The vector-add disassembly should include
RVV instructions such as `vsetivli`, `vle32.v`, `vfadd.vv`, and `vse32.v`.

## Failure classes and fixes

| Failure class | Typical symptom | Compatibility strategy |
| --- | --- | --- |
| Boolean ABI | `i1` pointer bitcasts fail pointer analysis | expose CPU boolean storage as `uint8` and compare with zero in the kernel |
| Unsupported collectives | unregistered `ttx.cumsum` or multi-result `tt.reduce` | use output-centric scans or scalar reductions for the covered sizes |
| Atomics | scalar or indirect `atomic_add` fails lowering | assign one program to each output and gather all of its contributions |
| Pointer analysis | pointer `select`, loop-carried pointers, or nested indirect loads fail | keep one base pointer per kernel and calculate integer offsets explicitly |
| Complex tensors | complex pointer loads/stores or lazy conjugate views fail | process real and imaginary storage explicitly, then rebuild the complex result |
| Layout and race correctness | wrong broadcast, reduction dimension, or colliding scatter writes | normalize shapes on the host and use deterministic output-centric kernels |
| Test configuration | invalid PyTorch reference shape or gradient construction | correct the reference while retaining the original coverage and assertions |

Several composite operators now reuse already validated primitives, for
example `stack`/`vstack` use `cat`, `tile` uses `repeat`, and the small `sort`
path uses a single-block sort. This keeps unsupported lowering patterns out of
the generated IR.

## Current limitations

The standalone QEMU runner accepts primitive scalar arguments and flat integer
or floating-point buffers. It does not yet serialize `fp16`, `bf16`, dynamic
input files, or complex shaped memrefs. Consequently, the complete PyTorch
FlagGems pytest process runs with the x86 CPU driver; QEMU validation is applied
to generated standalone kernels that cover the main lowering patterns above.
Native-RVV-only pytest cases remain guarded by their existing capability marker.
