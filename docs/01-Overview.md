# Project Overview

> **Audience:** beginners who have completed the Host CPU vector-add Quick
> Start, plus anyone who needs the project boundary before choosing a developer
> route.
>
> **Prerequisite:** [15-minute Quick Start](00-Getting-Started.md#15-minute-quick-start).
>
> **Outcome:** explain what Triton-RISCV owns and distinguish Host CPU execution
> from RISC-V ELF/QEMU execution. Compiler-pass details are intentionally left
> to the contributor guide.

Triton-RISCV connects Triton's Python kernel language to a CPU-oriented MLIR
and LLVM pipeline. Its primary target is RISC-V, but a host CPU mode makes the
same lowering path easier to develop, test, and debug on an `x86_64` machine.

This page answers four beginner questions:

1. What problem does the project solve?
2. Which parts come from Triton, Triton-RISCV, and Buddy?
3. What is the difference between host mode and RISC-V mode?
4. Where should I look in the repository?

## 1. The Problem in One Example

A Triton kernel describes many program instances that operate on blocks of
data:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

On a GPU backend, Triton maps the grid and operations to GPU concepts. On this
backend, Triton-RISCV must instead answer questions such as:

- What memory region does each pointer expression describe?
- Can a masked load become a regular, strided, or indexed CPU access?
- Which tensor operations can become Linalg and Vector operations?
- How are Triton's program IDs passed to a CPU function?
- Should the final object target the host CPU or `riscv64`?

The project implements those answers as MLIR analyses, conversion passes, a
Triton backend plugin, and CPU/RISC-V runtime helpers.

## 2. Who Owns Each Compiler Stage?

```text
Python source
  |
  |  Upstream Triton frontend
  v
TTIR
  |
  |  Triton-RISCV: triton-shared-opt and conversion passes
  v
Structured MLIR (Triton Structured / Linalg / Tensor / MemRef)
  |
  |  Buddy Compiler: bufferization, loops, and vector lowering
  v
Vector MLIR
  |
  |  Buddy + LLVM lowering
  v
LLVM IR
  |
  |  LLVM llc or Buddy llc
  v
Host object or RISC-V object
```

The boundaries are:

- **Upstream Triton** parses the Python AST, specializes `tl.constexpr`
  arguments, creates TTIR, and provides the JIT/runtime interfaces.
- **Triton-RISCV** analyzes Triton pointer expressions and masks, converts TTIR
  into structured forms, defines the CPU backend stages, and supplies launcher
  and RISC-V ELF helpers.
- **Buddy Compiler** supplies downstream transformations used for
  bufferization, vectorization, loop lowering, and some target-specific paths.
- **LLVM** translates LLVM-dialect MLIR, optimizes or lowers LLVM IR, and emits
  object code.

The `triton_shared` name remains in Python imports and binary names because the
project preserves compatibility with its origin in `triton-shared`.

## 3. The Two Main Execution Modes

### Host CPU mode

Use host mode while developing a kernel or a compiler pass. Python creates CPU
tensors, Triton compiles a native object for the current machine, and
`CPUDriver` builds and loads a launcher.

```python
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

The repository's `python/examples/conftest.py` does this automatically for
pytest. A standalone Python program must do it itself.

Host mode is useful for:

- fast correctness checks against PyTorch;
- debugging Python, TTIR, and MLIR lowering;
- running the examples on an `x86_64` developer machine;
- measuring host-side allocation and copy overhead.

It does **not** prove that the generated RISC-V instructions are correct or
fast.

### RISC-V ELF mode

Use RISC-V mode to validate target code generation. The helper API asks the
same backend to emit a `riscv64` object, generates a small C runner, links a
static Linux ELF, and optionally executes it with QEMU user-mode emulation.

```text
Triton kernel + Python test data
  -> RISC-V object
  -> generated C runner
  -> static ELF
  -> QEMU or real RISC-V Linux
```

This mode is useful for:

- checking that an ELF is a valid `ELF64 RISC-V` executable;
- verifying output without writing a runner by hand;
- inspecting RVV instructions such as vector loads, arithmetic, and stores;
- moving the resulting executable to compatible hardware.

QEMU is designed here for correctness and inspection. It is not a reliable
performance model for real hardware.

See [RISC-V ELF and QEMU](06-RISCV-QEMU.md) for the complete workflow.

## 4. Standalone IR Conversion

Compiler developers can run the middle layer without launching Python code.
`triton-shared-opt` accepts MLIR and exposes the project's conversion passes:

```sh
"$TRITON_SHARED_OPT_PATH" input.mlir \
  --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu \
  -o output.mlir
```

This is the fastest way to iterate on a single lowering rule because it avoids
the Python JIT, object generation, and runtime. Files under `test/` use this
style with lit and FileCheck.

## 5. Repository Map

| Path | What to look for |
| --- | --- |
| `backend/compiler.py` | active Triton stages and downstream Buddy/LLVM pass pipelines |
| `backend/driver.py` | CPU launcher generation, object loading, and runtime interface |
| `backend/riscv.py` | target settings, runner generation, ELF linking, QEMU, and disassembly |
| `backend/paths.py` | discovery rules for `triton-shared-opt`, Buddy, and LLVM tools |
| `include/triton-shared/Analysis/` | analysis interfaces used by conversions |
| `lib/Analysis/` | pointer/use/mask-related implementation |
| `include/triton-shared/Conversion/` | conversion pass declarations |
| `lib/Conversion/` | Triton-to-structured/Linalg/MemRef implementation |
| `lib/Dialect/` | project-specific intermediate dialects |
| `tools/triton-shared-opt/` | standalone optimizer registration and entry point |
| `python/examples/` | runnable kernels, CPU correctness tests, and RISC-V object tests |
| `python/performance/` | host CPU provider comparisons |
| `test/` | focused MLIR input/output regression tests |
| `scripts/` | build, environment, patch, and toolchain helpers |

## 6. What “Supported” Means

There is no single yes/no switch for an operator. A kernel is usable only when
all of these layers support its behavior:

1. upstream Triton can create valid TTIR;
2. pointer, mask, and use analyses can represent its memory accesses;
3. Triton-RISCV can convert every remaining operation;
4. Buddy/LLVM can lower the resulting structured and vector operations;
5. the CPU launcher or standalone runner can represent its arguments;
6. the output matches a reference for relevant shapes and edge cases.

That is why the examples and regression tests are more useful than a broad
feature claim. Start from the closest passing example, change one property at a
time, and keep a PyTorch reference beside it.

Known gaps include some non-contiguous pointer values across control flow,
legacy block-pointer patterns, float8/TF32 in the reference backend, and
standalone serialization for `fp16`/`bf16`. The detailed and evolving list is
in the compiler-contributor [Implementation details](02-Implementation.md#current-limitations) and
[Optimization opportunities](07-Optimization.md).

## 7. Optional Exploration for Kernel Developers

Beginners can stop here. The following IR exercise is useful only if you want
to inspect or change kernels.

After completing [Getting started](00-Getting-Started.md):

```sh
export TRITON_SHARED_DUMP_PATH="$PWD/artifacts/ir/overview-vec-add"
export TRITON_CACHE_DIR="$PWD/artifacts/cache/overview-vec-add"
rm -rf "$TRITON_SHARED_DUMP_PATH" "$TRITON_CACHE_DIR"
python -m pytest python/examples/test_vec_add.py -v -s
find "$TRITON_SHARED_DUMP_PATH" -maxdepth 1 -type f -print | sort
```

Open `tt.mlir`, `ttshared.mlir`, `vector.mlir`, `ll.mlir`, and `ll.ir` in that
order. Even without knowing every dialect, look for how `tt.load`, addition,
and `tt.store` become structured memory operations, vector operations, and
LLVM instructions. Success means pytest reports `PASSED` and all five files
appear. The [IR guide](03-IR.md) explains each file and the historical test's
non-asserting output behavior.

## Choose a Developer Route

- To write or port kernels, continue with
  [Operator migration](05-Operator-Migration.md).
- To validate RISC-V output, continue with the
  [minimal QEMU tutorial](06-RISCV-QEMU.md#minimal-risc-vqemu-tutorial).
- To change compiler passes, continue with the advanced
  [Implementation details](02-Implementation.md).

[Previous: Getting started](00-Getting-Started.md) · [Repository README](../README.md)
