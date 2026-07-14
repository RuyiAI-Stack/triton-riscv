# Triton-RISCV Documentation

The repository README has one entry point: the
[15-minute Quick Start](00-Getting-Started.md#15-minute-quick-start). Finish it
before choosing a longer route here.

If this is your first Triton, MLIR, or RISC-V project, follow only the Beginner
route. The other routes are references for specific kinds of development, not
additional prerequisites.

## Beginner Route

Goal: understand the project and run one correct kernel without reading compiler
internals.

1. Complete the
   **[15-minute Quick Start](00-Getting-Started.md#15-minute-quick-start)**.
   You will build the prebuilt configuration, run vector addition on the host
   CPU, and check its result against PyTorch.
2. Read **[What happened during vector add](00-Getting-Started.md#what-happened-during-vector-add)**
   in the same document. It explains the kernel, grid, mask, `CPUDriver`, and
   compiler stages in plain language.
3. Read **[Project overview](01-Overview.md)** for the project boundary and the
   difference between Host CPU and RISC-V execution.
4. Optional: when you have the cross-toolchain, complete the
   **[minimal RISC-V/QEMU tutorial](06-RISCV-QEMU.md#minimal-risc-vqemu-tutorial)**.

Stop there unless you want to write kernels or change the compiler.

If the Quick Start fails, use the first section of
[Debugging](04-Debug.md#vector-add-first-response) rather than reading every
compiler document.

## Kernel Developer Route

Goal: write or migrate Triton kernels and prove their behavior on the CPU and,
when required, RISC-V.

Prerequisite: complete the Beginner route through the project overview.

1. **[Operator migration](05-Operator-Migration.md)** — follow the vector-add
   case study from a GPU-oriented wrapper to a CPU test and RISC-V example.
2. **[Inspecting IR](03-IR.md)** — follow the same vector-add kernel through
   TTIR, structured/vector MLIR, LLVM IR, and object generation.
3. **[Debugging](04-Debug.md)** — distinguish frontend, lowering, host runtime,
   and QEMU failures.
4. **[RISC-V ELF and QEMU](06-RISCV-QEMU.md)** — add standalone target
   validation after Host CPU correctness is established.

Use [Benchmarking](08-Benchmark.md) only when correctness is stable. It is an
advanced host-development tool, not a beginner step and not a RISC-V hardware
measurement.

## Compiler Contributor Route

Goal: change analyses, MLIR conversions, backend stages, runtime code, or code
generation.

Prerequisite: complete the Quick Start and read the project overview.

1. **[Implementation details](02-Implementation.md)** — advanced compiler
   architecture, pointer/mask analysis, bufferization, ABI, and pass pipeline.
2. **[Inspecting IR](03-IR.md)** — identify the first stage where behavior
   changes.
3. **[Debugging](04-Debug.md)** — reduce failures and select the right test
   layer.
4. **[Optimization opportunities](07-Optimization.md)** — advanced roadmap for
   memory, vector, pointer, and FileCheck work.
5. **[Benchmarking](08-Benchmark.md)** — advanced host-side regression and
   performance measurement.

The last two documents are deliberately outside the Beginner route. Host
benchmark results do not predict performance on real RISC-V hardware.

## Document Catalog

| Document | Primary audience | Purpose |
| --- | --- | --- |
| [Getting started](00-Getting-Started.md) | Beginner | shortest Host CPU run, vector-add walkthrough, optional source build |
| [Project overview](01-Overview.md) | Everyone | project boundary and two execution routes |
| [Implementation details](02-Implementation.md) | Compiler Contributor | analyses, passes, bufferization, ABI, and runtime internals |
| [Inspecting IR](03-IR.md) | Kernel/Compiler Developer | inspect vector-add at each compiler stage |
| [Debugging](04-Debug.md) | Kernel/Compiler Developer | locate environment, frontend, lowering, runtime, or QEMU failures |
| [Operator migration](05-Operator-Migration.md) | Kernel Developer | complete migration case and correctness workflow |
| [RISC-V ELF and QEMU](06-RISCV-QEMU.md) | Kernel Developer | minimal target run plus advanced RISC-V reference |
| [Optimization opportunities](07-Optimization.md) | Compiler Contributor, advanced | optimization roadmap and code-shape expectations |
| [Benchmarking](08-Benchmark.md) | Kernel/Compiler Developer, advanced | host CPU comparisons; not hardware performance |

## Glossary

Terms are introduced where they are first needed. This table is a reference,
not a prerequisite reading list.

| Term | Intuitive meaning | Precise meaning here |
| --- | --- | --- |
| Triton kernel | a Python function describing block-wise parallel work | a function decorated with `@triton.jit` and compiled by Triton's frontend |
| Backend | the compiler path for a target | Triton's plugin that defines compiler stages and emits an object file |
| Driver | the code that starts a compiled kernel | the runtime adapter that loads a host object and launches program instances |
| TTIR | the first compiler-readable form of the kernel | Triton IR, an MLIR dialect/module produced from the specialized Python kernel |
| Linalg | structured tensor/loop computation | an MLIR dialect used for operations such as elementwise work, reductions, and matmul |
| MemRef | a shaped view of memory | an MLIR type/dialect carrying element type, shape, offset, and stride information |
| Vector Dialect | explicit portable vector operations | MLIR operations that Buddy later lowers toward LLVM vectors or target instructions |
| LLVM IR | low-level code before machine code | textual LLVM intermediate representation consumed by `llc`/Buddy LLVM tools |
| ABI | the calling agreement between generated pieces | argument types/order and launch values shared by the kernel and its host or generated runner |
| Host mode | run on the machine executing Python | compile a native object and launch it through `CPUDriver` |
| Cross-compile mode | generate code for another machine | emit a RISC-V object/ELF for QEMU or compatible RISC-V Linux hardware |
| RVV | RISC-V vector instructions | the RISC-V Vector Extension used by target vector code |
| lit/FileCheck | compiler text regression tests | LLVM's test runner and pattern checker used under `test/` |

Some paths and imports retain the historical `triton_shared` name. For example,
the backend package is `triton.backends.triton_shared`, and the optimizer is
`triton-shared-opt`. User-facing prose uses Triton-RISCV.

## Source of Truth

The project changes quickly. When prose and code disagree, prefer:

1. `backend/compiler.py` for active stages and pass pipelines;
2. `backend/riscv.py` for RISC-V arguments, target settings, ELF, and QEMU;
3. `scripts/build.sh` and `scripts/triton-riscv-env.sh` for build layouts;
4. `.github/workflows/ci.yml` for currently tested hosts and commands;
5. `python/examples/`, `python/performance/`, and `test/` for actual behavior;
6. these guides.

[Back to the repository README](../README.md)
