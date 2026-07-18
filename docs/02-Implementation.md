# Implementation Details

> **Audience: Compiler Contributors.** This is not a beginner tutorial and is
> not required to write or run the first kernel.
>
> **Prerequisites:** complete the
> [15-minute Quick Start](00-Getting-Started.md#15-minute-quick-start), read the
> [project overview](01-Overview.md), and be comfortable reading MLIR.
>
> **Outcome:** follow one kernel from TTIR to a callable object, identify which
> analysis or pass owns a transformation, and choose the right regression test.

This document explains the compiler's internal analyses, representations, pass
pipeline, runtime calling convention, and current limitations. Kernel
developers should normally read [Operator migration](05-Operator-Migration.md),
[Inspecting IR](03-IR.md), and [Debugging](04-Debug.md) instead.

For a conceptual introduction first, read [Project overview](01-Overview.md).

## Active Backend Stages

`CPUBackend.add_stages` in `backend/compiler.py` defines the source of truth for
the normal compilation path:

| Triton stage key | Representation | Main implementation |
| --- | --- | --- |
| `ttir` | Triton MLIR after common TTIR cleanup | `CPUBackend.make_ttir` |
| `ttsharedir` | structured/Linalg/MemRef-oriented MLIR | `_ttir_to_ttsharedir` and `triton-shared-opt` |
| `vectorir` | Buddy vector-oriented MLIR | `_ttsharedir_to_vectorir` |
| `llir` | textual LLVM IR | `_vectorir_to_llir` |
| `obj` | host or RISC-V object bytes | `_llir_to_bin` |

The optional IME matrix-extension path skips the separate `vectorir` stage and
uses a Buddy-specific pipeline. It is enabled with
`TRITON_RISCV_USE_IME=1` and should be treated as an advanced target-specific
mode, not the default beginner path.

## TTIR Preparation

Upstream Triton turns the Python AST into TTIR. Triton-RISCV then runs a short
set of upstream TTIR passes before handing the module to its middle layer:

- inline Triton functions;
- rewrite current tensor descriptors to pointers;
- canonicalize and combine operations;
- reorder broadcasts and eliminate common subexpressions;
- move loop-invariant operations and unroll eligible loops;
- remove dead symbols.

At this point the program still contains Triton operations such as `tt.load`,
`tt.store`, `tt.addptr`, `tt.reduce`, and `tt.dot`. The difficult part is not
the arithmetic; it is recovering the memory-access structure encoded by
pointer expressions, shapes, masks, and control flow.

## Memory Access Analysis

Triton permits pointer tensors and arbitrary pointer arithmetic. Structured
MLIR operations prefer explicit shapes, offsets, sizes, and strides. The
middle layer therefore analyzes each memory access before lowering it.

### Pointer analysis

Pointer analysis tracks how a base pointer is transformed by operations such
as splat, range creation, broadcast, expand-dims, and `tt.addptr`. For a regular
access, it tries to recover a view like:

```text
base pointer + offset
shape  = [rows, columns]
stride = [row_stride, column_stride]
```

That information can become `memref.reinterpret_cast`, a subview, or another
structured memory representation. If addresses are data-dependent or unrelated
to one another, the access must stay on an unstructured gather/scatter path.

### Use analysis

Address calculations often share operations with ordinary data calculations.
Use analysis classifies values so address-only operations can be removed after
their meaning has been captured:

- **MetaUse** — used only to calculate an address or memory shape;
- **MixedUse** — used both by address calculation and by the kernel's data
  computation.

A mixed-use value may need to be cloned so the metadata path can be rewritten
without changing the data path.

### Mask analysis

Mask analysis determines how a load/store mask relates to the access shape.
Common boundary masks can often become sizes, bounds, or structured slices.
Data-dependent masks may require an unstructured representation or explicit
predication.

These analyses are deliberately conservative. Rejecting an unsupported pattern
is safer than silently losing an offset, stride, branch-local value, or mask.

## The Experimental Structured Pipeline

The active backend invokes:

```sh
triton-shared-opt input.mlir \
  --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu
```

The composite pass is implemented in
`lib/Conversion/TritonToLinalgExperimental/`. At a high level it performs:

1. **Triton to Triton Structured** — represent regular pointer/mask state with
   project-specific structured operations.
2. **Triton to Unstructured** — isolate accesses that cannot use the regular
   structured path, including gather/scatter cases.
3. **Triton arithmetic to Linalg** — express elementwise tensor computation as
   Linalg where possible.
4. **Structured and unstructured memory conversion** — materialize MemRef,
   Tensor, affine, or indexed memory operations.
5. **Pointer cleanup** — convert remaining Triton pointers, reconcile casts,
   remove dead metadata values, and canonicalize the result.

The output is called `ttsharedir` by the backend. It is not one dialect; it is
a legal mix of standard MLIR dialects and, where needed, project-specific
intermediate operations.

## Common Operation Mappings

The exact result depends on shape, mask, and aliasing information, but these are
the intended mappings:

| Triton operation or pattern | Typical structured result |
| --- | --- |
| `tt.load` with regular access | MemRef view plus tensor/vector load path |
| `tt.store` with regular access | destination MemRef view and store/materialization |
| elementwise `arith`/`math` | `linalg.generic` or equivalent vector/loop operations |
| `tt.reduce` | `linalg.reduce` for supported reduction bodies |
| `tt.dot` | `linalg.matmul` or a related structured contraction |
| regular boundary mask | bounded size/slice or predicated access |
| data-dependent addresses | gather/scatter or explicit indexed/loop path |

A simple regular load may initially look like this after conversion:

```mlir
%view = memref.reinterpret_cast %base
  to offset: [%offset], sizes: [%size], strides: [1]
%tmp = memref.alloc() : memref<...>
memref.copy %view, %tmp
%tensor = bufferization.to_tensor %tmp restrict writable
```

This is semantically useful because the shape and stride are explicit, but the
temporary allocation and copy are expensive. Removing them safely is one of
the main optimization goals described in
[Optimization opportunities](07-Optimization.md).

Stores similarly aim to connect a computed tensor to the real output view. If
the result is built in a separate allocation first, bufferization may introduce
an avoidable final copy. Destination-passing transformations try to compute
directly into the output when aliasing, mask, and shape rules allow it.

## Bufferization, Vectorization, and LLVM

**Bufferization** turns value-like tensor operations into operations over
concrete memory buffers. **Vectorization** exposes groups of scalar operations
as Vector Dialect operations that can later become target vector code.

After the Triton-specific conversion, Buddy handles the generic lowering work.
The default pipeline in `backend/compiler.py` includes these broad phases:

1. convert empty tensors to allocation-backed tensors;
2. one-shot bufferization and buffer deallocation;
3. lower supported Linalg operations to Buddy's vector IR;
4. canonicalize and eliminate common subexpressions;
5. lower vector, affine, SCF, arithmetic, math, MemRef, and function operations
   to LLVM-compatible forms;
6. translate LLVM-dialect MLIR to textual LLVM IR.

The final `llc` invocation emits position-independent object code. Normally it
targets the current host. Cross-compilation supplies a RISC-V triple and feature
set through `CPUOptions` and `RiscvToolchain`.

## Kernel ABI and Runtime

An **ABI** (application binary interface) is the exact calling agreement
between separately generated pieces: argument types and order, return behavior,
and the extra launch values passed to the kernel.

The lowered kernel receives its original non-`constexpr` arguments plus six
32-bit launch values:

```text
grid_x, grid_y, grid_z, program_id_x, program_id_y, program_id_z
```

These values implement `tl.num_programs`/launch-grid context and
`tl.program_id` without a GPU runtime.

### Host launcher

`backend/driver.py` generates a C++ extension that:

1. converts Python/Torch arguments to the kernel ABI;
2. loads the object produced by the backend;
3. invokes the kernel for the requested program IDs;
4. exposes the result through Triton's normal launch interface.

The driver is intentionally not selected automatically:

```python
triton.runtime.driver.set_active(CPUDriver())
```

### Standalone RISC-V runner

`backend/riscv.py` provides a separate ahead-of-time flow. It embeds Python
lists or supported array values into generated C source, calls the kernel for
every point in a one-, two-, or three-dimensional grid, optionally verifies
outputs, links a static ELF, and launches QEMU.

The host driver and standalone runner share compiler stages but solve different
runtime problems. A kernel passing in host mode may still expose target or
serialization issues in standalone RISC-V mode.

## Current Limitations

Support is test-driven and evolves with the pinned Triton and Buddy revisions.
Important current boundaries include:

- pointer analysis does not represent every arbitrary address calculation;
- some pointer tensors crossing `scf.if` or loop yields remain unsupported;
- several legacy/wraparound/block-pointer regression cases are still `XFAIL`;
- the reference CPU backend does not enable float8 or TF32;
- reductions and math operations are supported incrementally rather than as a
  complete Triton language matrix;
- automatic standalone-runner serialization supports common integer, `fp32`,
  and `fp64` arguments, but not `fp16`, `bf16`, complex MemRef shapes, or
  dynamic input files;
- host CPU execution and QEMU correctness do not establish performance on real
  RISC-V hardware.

Use these files as the most precise compatibility evidence:

- `python/examples/conftest.py` for upstream language cases enabled on CPU;
- `python/examples/test_*.py` for end-to-end kernel behavior;
- `test/**/*.mlir` for individual conversion rules and `XFAIL` cases;
- `backend/riscv.py` for standalone argument types and target settings.

## Testing a Compiler Change

Use the smallest test that owns the behavior, then widen coverage:

```sh
# Focus the configured lit suite by test-name regex
"$LLVM_BINARY_DIR/llvm-lit" -sv \
  "$BUILD_DIR/third_party/triton_shared/test" \
  --filter='masked_ldst'

# Entire compiler regression suite
cmake --build "$BUILD_DIR" --target check-triton-shared-lit-tests

# One end-to-end Python kernel
python -m pytest python/examples/test_vec_add.py -v

# Common example suite
python -m pytest python/examples/ \
  --ignore=python/examples/test_core.py \
  --ignore=python/examples/test_annotations.py \
  --ignore=python/examples/flaggems \
  -v
```

The configured lit directory is generated inside the Triton build tree. Change
the `--filter` regular expression to part of the test filename or test name. If
direct `llvm-lit` invocation is inconvenient, use the CMake target while
iterating.

Success criteria are explicit: lit prints no `FAIL`/`XPASS` result and exits
with status 0; the CMake target finishes successfully; and pytest ends with
`PASSED`. Here `test_vec_add.py` is a compilation/launch smoke test because its
historical body only prints the numerical difference. Use the strict assertion
from the [Quick Start](00-Getting-Started.md#step-7-run-and-strictly-check-host-cpu-vector-add)
when numerical correctness is part of the change.

For an optimization, test both positive and negative behavior: prove that the
desired vector/store form appears, and prove that unsafe aliasing, mask, stride,
or control-flow cases do not take the optimization.

## Where to Make a Change

| Change | Start here |
| --- | --- |
| Add a pointer/mask fact | `include/triton-shared/Analysis/`, `lib/Analysis/` |
| Change structured TTIR conversion | `lib/Conversion/TritonToStructured/` |
| Change unstructured/gather-scatter conversion | `lib/Conversion/TritonToUnstructured/`, `lib/Conversion/UnstructuredToMemref/` |
| Change elementwise Linalg conversion | `lib/Conversion/TritonArithToLinalg/` |
| Change the composite pipeline | `lib/Conversion/TritonToLinalgExperimental/` |
| Change downstream pass order | `backend/compiler.py` |
| Change host launch semantics | `backend/driver.py` |
| Change RISC-V target/link/QEMU behavior | `backend/riscv.py` |
| Add a pass-level regression | `test/` |
| Add end-to-end coverage | `python/examples/` |

[Previous: Project overview](01-Overview.md) · [Repository README](../README.md) · [Next: Inspecting IR](03-IR.md)
