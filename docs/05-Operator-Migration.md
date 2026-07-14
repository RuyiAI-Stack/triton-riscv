# Operator Migration Guide

> **Audience:** Kernel Developers adapting an existing Triton kernel to the
> Host CPU backend and optionally to RISC-V.
>
> **Prerequisites:** complete the
> [15-minute Quick Start](00-Getting-Started.md#15-minute-quick-start) and read
> the [project overview](01-Overview.md).
>
> **Outcome:** reduce a GPU-oriented operator to a supported contract, add a
> real correctness assertion, validate Host CPU execution, optionally validate
> a RISC-V ELF, and identify the failing layer when migration stops.

Migration means more than making a file compile. The result must preserve the
claimed operator semantics and pass a reference check for relevant shapes,
dtypes, strides, aliases, and edge cases.

## Complete Case Study: Vector Add

The repository contains three useful versions of the same idea:

| File | Role |
| --- | --- |
| `python/examples/test_vec_add.py` | tutorial-style kernel and Host CPU pytest example, with some inherited GPU-oriented comments |
| `python/performance/test_performance_vec_add.py` | explicit `CPUDriver`, CPU inputs, strict provider comparison, and timing |
| `python/examples/rvv_vec_add_elf.py` | standalone RISC-V ELF generation and QEMU verification |

Together they show the migration sequence: preserve the Triton computation,
replace GPU runtime assumptions, prove Host CPU correctness, then add a separate
RISC-V target test.

### Step 1: Write Down the Supported Contract

Start narrow:

```text
operation: out[i] = x[i] + y[i]
dtypes: float32
shape: equal-length 1-D tensors
layout: contiguous
device for Host test: CPU
tail behavior: length may not be a multiple of BLOCK_SIZE
aliasing: output is newly allocated
```

This contract tells reviewers what is implemented and tells the wrapper what
to reject. Passing one contiguous `float32` case does not imply support for
every dtype, strided view, or in-place alias.

### Step 2: Keep the Target-Neutral Kernel Core

The core kernel does not need CUDA-specific APIs:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

Why this is a good first migration:

- `tl.program_id` and the grid divide work into blocks without naming a GPU;
- `tl.arange` creates regular per-block offsets the pointer analysis can
  understand;
- the tail mask makes non-multiple lengths safe;
- load, elementwise add, and store have existing Host CPU and RISC-V coverage.

Do not interpret this as a general support claim for all pointer expressions or
all Triton operations. The examples and regression tests are the evidence for
specific patterns.

### Step 3: Replace the GPU Runtime Assumption

A GPU-oriented wrapper commonly creates CUDA tensors, assumes a GPU driver, or
contains CUDA synchronization and device guards. The Host CPU path must select
the repository driver explicitly:

```python
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
```

`CPUDriver` is the runtime adapter that loads and launches the native Host CPU
object. Pytest files under `python/examples/` get this automatically from
`conftest.py`; standalone scripts and performance scripts select it themselves.

Use CPU tensors and validate the contract in the wrapper:

```python
def run_vec_add(x, y):
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("run_vec_add requires CPU tensors")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("run_vec_add requires equal-length 1-D tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("run_vec_add currently requires contiguous tensors")

    output = torch.empty_like(x)
    n_elements = output.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

The repository's performance example uses assertions for its internal test
inputs rather than public `ValueError` checks, but the supported shape contract
is the same.

### Step 4: Add a Real Correctness Test

Use PyTorch only as an independent reference:

```python
def test_vec_add_cpu():
    triton.runtime.driver.set_active(CPUDriver())
    torch.manual_seed(0)
    x = torch.rand(257, device="cpu", dtype=torch.float32)
    y = torch.rand(257, device="cpu", dtype=torch.float32)
    expected = x + y
    actual = run_vec_add(x, y)
    torch.testing.assert_close(actual, expected)
```

The size 257 is intentional: with `BLOCK_SIZE=1024`, most lanes are masked and
the tail path is exercised. Add an exact block multiple and a multi-block case
as separate tests.

In `python/performance/test_performance_vec_add.py`, the repository uses:

```python
benchmark.compare_providers(
    "bench_vec_add(...)",
    {
        "torch": lambda: x + y,
        "triton-riscv": lambda: run_vec_add(x, y),
    },
)
```

`compare_providers` runs the PyTorch callable for expected output, runs the
Triton-RISCV callable, verifies that both return CPU tensors, and calls its
close-check helper before timing. The PyTorch callable is a baseline, **not a
fallback and not code compiled by Triton-RISCV**. An optional `torch+buddy-mlir`
provider is a separate Buddy `torch.compile` path.

Run the repository comparison after the small test passes:

```sh
python python/performance/run_all.py vec_add --torch-threads 1
```

Success criteria:

- the command exits with status 0;
- the `triton-riscv` status is `PASS` for every vector-add row;
- the CSV is written to `artifacts/performance/triton_riscv_bench.csv`.

Timing values are host measurements and are irrelevant to migration
correctness. They do not predict RISC-V hardware performance.

### Step 5: Add a Separate RISC-V Target Case

Do not treat the Host CPU test as a QEMU test. The RISC-V example uses
`standalone_kernel_cli` with Python lists that are embedded in an automatically
generated C runner:

```python
standalone_kernel_cli(
    vector_add_kernel,
    arguments={
        "x_ptr": x,
        "y_ptr": y,
        "output_ptr": [0.0] * size,
        "n_elements": size,
    },
    constexprs={"BLOCK_SIZE": block_size},
    grid=(triton.cdiv(size, block_size),),
    expected={"output_ptr": [a + b for a, b in zip(x, y)]},
    default_output="artifacts/riscv/rvv-vector-add.elf",
)
```

The `expected` mapping generates checks inside the RISC-V runner. With the
cross-toolchain installed:

```sh
python python/examples/rvv_vec_add_elf.py
```

Success prints `PASS` and the ELF path. Follow the
[minimal RISC-V/QEMU tutorial](06-RISCV-QEMU.md#minimal-risc-vqemu-tutorial)
for toolchain checks and exact output.

### Step 6: Locate a Migration Failure

Use the same kernel and change one property at a time. The first missing or bad
stage identifies the owner:

| Failure point | Evidence | Likely area |
| --- | --- | --- |
| Frontend | no `tt.mlir`; Python/Triton rejects syntax, type, or specialization | kernel syntax, signature, `tl.constexpr`, upstream Triton support |
| Triton-RISCV lowering | `tt.mlir` exists; `ttshared.mlir` is missing or wrong | pointer/mask/use analysis or Triton-to-structured conversion |
| Buddy/LLVM lowering | `ttshared.mlir` exists; `vector.mlir`, `ll.mlir`, or `ll.ir` fails | bufferization, vector, standard MLIR, or LLVM conversion |
| Host runtime | `ll.ir`/object exists; Python launch crashes or arguments are wrong | `CPUDriver`, launcher ABI, bounds, or kernel memory behavior |
| Numerical correctness | Host launch completes; PyTorch comparison fails | kernel semantics, grid, mask, strides, dtype, aliasing, or lowering bug |
| RISC-V link | Host passes; ELF is not produced | target triple, ISA features, ABI, compiler, or sysroot |
| QEMU/target | ELF exists; QEMU traps or runner says verification failed | target code, QEMU CPU features, runner arguments/data, or target-only bug |

The **ABI** is the calling agreement between the generated kernel and launcher:
argument types/order plus grid and program-ID values. An ABI mismatch is a
runtime or target integration problem, not a Python reference failure.

## General Migration Workflow

After the vector-add case, apply this sequence to a real operator.

### Choose the Smallest Useful Kernel

An operator file may contain several kernels, autotuning tables, device
dispatch, imported helpers, and fallbacks. Start with one representative kernel
and one explicit input class. Good first candidates have regular contiguous
access and simple elementwise, reduction, or matmul structure.

Find the closest passing example:

| Kernel shape | Useful starting files |
| --- | --- |
| elementwise and masked tail | `test_vec_add.py`, `test_mask.py` |
| reductions | `test_reduce.py`, `test_softmax.py` |
| matrix multiplication | `test_matmul.py`, `test_mm.py` |
| normalization | `test_layernorm.py`, `test_norm_layernorm.py` |
| non-contiguous or indexed access | `test_addptr.py`, `test_gather_scatter.py` |
| early return or loops | `test_early_return.py`, `test_nested_loops.py` |
| RISC-V standalone output | `rvv_vec_add_elf.py` |

### Remove GPU-Specific Control Code Carefully

- replace `@triton.autotune` with one known configuration while porting;
- replace GPU heuristics with explicit `tl.constexpr` values;
- remove CUDA-only device/stream/synchronization calls from the Host wrapper;
- use CPU tensors and select `CPUDriver`;
- use `input_precision="ieee"` where GPU-specific TF32 was assumed;
- keep only helpers required by the selected kernel.

If a GPU feature contributes to semantics, do one of the following:

1. implement an equivalent supported formulation;
2. narrow the wrapper contract and raise a clear error;
3. document the case as unsupported and add a regression test.

Never delete masking, bounds checks, dtype conversion, ordering, or a fallback
merely to make compilation succeed.

### Inline Dependencies Deliberately

Inlining a small Triton helper can make a focused example self-contained. List
every external symbol first, preserve `tl.constexpr` specialization and dtype
casts, and add a reference test for boundary behavior. Do not duplicate a large
utility module into every migrated operator.

### Expand the Correctness Matrix

At minimum, cover:

- a small shape;
- an exact block multiple;
- a masked non-multiple;
- every claimed dtype;
- non-contiguous input or an explicit rejection;
- zero-sized/scalar behavior if the public API permits it;
- aliasing or in-place behavior if promised;
- negative, zero, large, NaN, and infinity values where meaningful;
- sensitive reduction/math cases with justified tolerances.

Use deterministic seeds and `torch.testing.assert_close`. A broad tolerance
must come from the dtype or algorithm, not from an unexplained wrong access.

## Common Porting Problems

### Autotuning Assumes a GPU Model

Start with one configuration. CPU and RISC-V tradeoffs differ from GPU warps,
stages, and shared memory, so retaining a CUDA tuning table does not validate it
for this backend.

### A `tl.math` Function Is Unavailable

Check current examples and conversion tests. Some migrated FlagGems files
document missing functions such as `acos` or `atan`. Use a tested approximation
only if it satisfies the numerical contract; otherwise reject the operation.

### Pointer Lowering Fails on a View

Reduce the case to base pointer, shape, offset, and stride. Test contiguous and
non-contiguous forms separately. A gather/scatter access is not a contiguous
load merely because one input happens to produce the same values.

### Control Flow Yields Pointer Tensors

Some pointer state across `scf.if` or loops is a known limitation. Simplify the
flow only when semantics permit; otherwise add a focused regression/XFAIL and
fix the representation rather than dropping a branch-local offset.

### The Test Prints an Error but Still Passes

Replace diagnostics with assertions. `python/examples/test_vec_add.py` is an
example of a historical test that prints maximum error without asserting it;
the Quick Start adds the strict check externally.

## Review Checklist

- [ ] The supported input contract is documented.
- [ ] The Host wrapper selects `CPUDriver` and uses CPU tensors.
- [ ] Unsupported shapes, strides, dtypes, or aliases are rejected clearly.
- [ ] No required semantics disappeared with GPU-specific control code.
- [ ] Correctness is asserted against an independent reference.
- [ ] Block boundary, dtype, stride, alias, and edge cases are covered as
      claimed.
- [ ] A fresh-cache compile succeeds.
- [ ] The first bad IR stage is understood for any remaining failure.
- [ ] RISC-V/QEMU coverage exists only when target execution is claimed.
- [ ] PyTorch reference/fallback behavior is not presented as compiler support.
- [ ] Host benchmark numbers are not presented as RISC-V hardware results.

Migrated FlagGems examples live under `python/examples/flaggems/`. Keep a small
end-to-end correctness test beside each supported contract so Triton, Buddy,
and MLIR updates cannot silently break it.

[Previous: Debugging](04-Debug.md) · [Documentation routes](README.md) · [Next: RISC-V ELF and QEMU](06-RISCV-QEMU.md)
