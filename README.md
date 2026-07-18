# Triton-RISCV

Triton-RISCV is an experimental compiler backend that lets Triton kernels run
on a host CPU or be cross-compiled for RISC-V. You still write a kernel in
Python with `@triton.jit`; this project supplies the CPU-oriented compiler path,
runtime support, RISC-V ELF helpers, tests, and examples.

> **Project status:** Triton-RISCV supports a useful and growing subset of
> Triton, but it is not a drop-in replacement for every CUDA or ROCm kernel.
> Start with the included vector-add example before migrating a larger operator.

## Start Here

There is one recommended first step:

**Follow the [15-minute Quick Start](docs/00-Getting-Started.md#15-minute-quick-start).**

It builds the CI-tested prebuilt-toolchain configuration, runs vector addition,
and performs an explicit result check against PyTorch.

The first example uses the **Host CPU backend**: it compiles a native object for
the machine running Python and launches it through `CPUDriver`. It does **not**
run RISC-V code or QEMU. After that succeeds, the same kernel can be taken down
the separate [RISC-V ELF and QEMU path](docs/06-RISCV-QEMU.md).

Do not begin with the source build, compiler implementation, optimization, or
benchmark documents unless that is already your goal. The
[documentation paths](#documentation-paths) provide separate routes for beginners,
kernel developers, and compiler contributors.

## What the Project Does

**Triton** is a Python language for writing data-parallel kernels.
**MLIR** is the compiler framework used to represent and transform those
kernels in several intermediate forms. A **backend** is the part of Triton that
turns those forms into code for a target; a **driver** is the runtime adapter
that loads and launches the generated code.

The normal compiler path is:

```text
Python @triton.jit kernel
  -> TTIR                         Triton's first compiler form
  -> Structured / Vector MLIR    explicit computation and memory operations
  -> LLVM IR                     low-level, target-ready program
  -> object file                 host CPU object or RISC-V object
```

TTIR means **Triton IR**, the MLIR form produced from the Python kernel.
Structured MLIR uses operations such as Linalg for tensor/loop computation and
MemRef for shaped memory views. Vector MLIR makes vector operations explicit.
LLVM IR is the low-level representation consumed by LLVM code generation.
These terms are introduced gradually in the Quick Start and collected in the
[glossary](#glossary).

Upstream Triton owns the Python frontend and initial TTIR passes.
Triton-RISCV owns the CPU backend plugin, Triton-to-structured lowering,
launcher, RISC-V helpers, and project tests. Buddy Compiler and LLVM provide
the downstream lowering and machine-code tools. No NVIDIA or AMD toolchain is
required.

The code began as a fork of
[`microsoft/triton-shared`](https://github.com/microsoft/triton-shared). The
historical `triton_shared` name remains in Python imports and binary paths for
compatibility.

## Documentation Paths

Complete the [15-minute Quick Start](docs/00-Getting-Started.md#15-minute-quick-start)
before choosing a longer route. If this is your first Triton, MLIR, or RISC-V
project, follow only the Beginner path.

### Beginner

Goal: understand the project and run one correct kernel without reading
compiler internals.

1. Complete the
   [15-minute Quick Start](docs/00-Getting-Started.md#15-minute-quick-start).
2. Read
   [What happened during vector add](docs/00-Getting-Started.md#what-happened-during-vector-add)
   to understand the kernel, grid, mask, `CPUDriver`, and compiler stages.
3. Read the [project overview](docs/01-Overview.md) for the project boundary and
   the difference between Host CPU and RISC-V execution.
4. Optional: when the cross-toolchain is available, complete the
   [minimal RISC-V/QEMU tutorial](docs/06-RISCV-QEMU.md#minimal-risc-vqemu-tutorial).

Stop there unless you want to write kernels or change the compiler. If the
Quick Start fails, begin with
[Vector-Add First Response](docs/04-Debug.md#vector-add-first-response).

### Kernel Developer

Goal: write or migrate Triton kernels and prove their behavior on the Host CPU
and, when required, RISC-V.

1. [Operator migration](docs/05-Operator-Migration.md) — follow the vector-add
   case from a GPU-oriented wrapper to a CPU test and RISC-V example.
2. [Inspecting IR](docs/03-IR.md) — follow the same kernel through TTIR,
   structured/vector MLIR, LLVM IR, and object generation.
3. [Debugging](docs/04-Debug.md) — distinguish frontend, lowering, host
   runtime, and QEMU failures.
4. [RISC-V ELF and QEMU](docs/06-RISCV-QEMU.md) — add target validation after
   Host CPU correctness is established.

Use [Benchmarking](docs/08-Benchmark.md) only after correctness is stable. It
measures the host development path, not real RISC-V hardware performance.

### Compiler Contributor

Goal: change analyses, MLIR conversions, backend stages, runtime code, or code
generation.

1. [Implementation details](docs/02-Implementation.md) — advanced compiler
   architecture, pointer/mask analysis, bufferization, ABI, and pass pipeline.
2. [Inspecting IR](docs/03-IR.md) — identify the first stage where behavior
   changes.
3. [Debugging](docs/04-Debug.md) — reduce failures and select the right test
   layer.
4. [Optimization opportunities](docs/07-Optimization.md) — advanced roadmap
   for memory, vector, pointer, and FileCheck work.
5. [Benchmarking](docs/08-Benchmark.md) — advanced host-side regression and
   performance measurement.

The implementation, optimization, and benchmark guides are deliberately
outside the Beginner path.

## Documentation Map

| Document | Primary audience | Purpose |
| --- | --- | --- |
| [Getting started](docs/00-Getting-Started.md) | Beginner | shortest Host CPU run, vector-add walkthrough, optional source build |
| [Project overview](docs/01-Overview.md) | Everyone | project boundary and two execution routes |
| [Implementation details](docs/02-Implementation.md) | Compiler Contributor | analyses, passes, bufferization, ABI, and runtime internals |
| [Inspecting IR](docs/03-IR.md) | Kernel/Compiler Developer | inspect vector-add at each compiler stage |
| [Debugging](docs/04-Debug.md) | Kernel/Compiler Developer | locate environment, frontend, lowering, runtime, or QEMU failures |
| [Operator migration](docs/05-Operator-Migration.md) | Kernel Developer | complete migration case and correctness workflow |
| [RISC-V ELF and QEMU](docs/06-RISCV-QEMU.md) | Kernel Developer | minimal target run plus advanced RISC-V reference |
| [Optimization opportunities](docs/07-Optimization.md) | Compiler Contributor, advanced | optimization roadmap and code-shape expectations |
| [Benchmarking](docs/08-Benchmark.md) | Kernel/Compiler Developer, advanced | host CPU comparisons; not hardware performance |

## Glossary

| Term | Intuitive meaning | Precise meaning here |
| --- | --- | --- |
| Triton kernel | a Python function describing block-wise parallel work | a function decorated with `@triton.jit` and compiled by Triton's frontend |
| Backend | the compiler path for a target | Triton's plugin that defines compiler stages and emits an object file |
| Driver | the code that starts a compiled kernel | the runtime adapter that loads a host object and launches program instances |
| TTIR | the first compiler-readable form of the kernel | Triton IR produced from the specialized Python kernel |
| Linalg | structured tensor/loop computation | an MLIR dialect for elementwise work, reductions, matmul, and related operations |
| MemRef | a shaped view of memory | an MLIR type carrying element type, shape, offset, and stride information |
| Vector Dialect | explicit portable vector operations | MLIR operations later lowered toward LLVM vectors or target instructions |
| LLVM IR | low-level code before machine code | the representation consumed by LLVM object-code generation |
| ABI | the calling agreement between generated pieces | argument types/order and launch values shared by the kernel and its launcher |
| Host mode | run on the machine executing Python | compile a native object and launch it through `CPUDriver` |
| Cross-compile mode | generate code for another machine | emit a RISC-V object/ELF for QEMU or compatible hardware |
| RVV | RISC-V vector instructions | the RISC-V Vector Extension used by target vector code |
| lit/FileCheck | compiler text regression tests | LLVM's test runner and pattern checker used under `test/` |

## Repository Map

```text
backend/             Triton CPU backend, driver, and RISC-V ELF helpers
include/, lib/       MLIR analyses, dialects, conversions, and transforms
tools/               triton-shared-opt command-line optimizer
python/examples/     runnable kernels and correctness-oriented examples
python/performance/  host CPU provider comparisons
test/                lit/FileCheck compiler regression tests
scripts/             build, environment, patch, and toolchain helpers
docs/                user and contributor guides
patches/             patches applied to the pinned upstream Triton checkout
triton-hash.txt      exact compatible upstream Triton commit
```

## Important Limitations

- The documented build workflows target Linux. Other hosts may need manual
  toolchain and launcher changes.
- The CPU driver must be selected explicitly outside the repository's pytest
  configuration.
- Triton features are supported incrementally. Float8 and TF32 are not enabled
  by the reference CPU backend, and some pointer/control-flow patterns remain
  unsupported.
- A PyTorch result used as a test reference does not mean that Triton-RISCV
  compiled that PyTorch operation or provides a fallback for it.
- Host CPU benchmarks measure the host compiler/runtime path. They do not
  predict performance on real RISC-V hardware.
- QEMU is primarily a correctness and code-generation tool, not a hardware
  performance simulator.

See the advanced [implementation limitations](docs/02-Implementation.md#current-limitations)
and [optimization roadmap](docs/07-Optimization.md) for technical detail.

## Getting Help and Contributing

If a build or kernel fails, follow the vector-add triage flow in the
[debugging guide](docs/04-Debug.md). Include the exact command, host and Python
versions, relevant environment paths, last successful IR stage, and a minimal
kernel when reporting a problem.

Contributions to code, tests, examples, and documentation are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).
