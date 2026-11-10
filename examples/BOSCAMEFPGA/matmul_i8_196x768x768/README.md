# matmul_i8_196x768x768 — Triton INT8 GEMM on the NH/E6 FPGA

The Triton frontend is provided by the external `triton-riscv` AOT exporter.
See [the Contract handoff guide](../../../docs/BoscameAOT.md) for installation,
tool selection, the API and migration of old commands.

This example lowers `A[196,768] i8 @ B[768,768] i8ᵀ + C[196,768] i32`
through the experimental BOSCAME backend.  B is physically `[N,K]`, with K
contiguous.  The Triton source is [kernels.py](kernels.py); the global
compile-time constants are `M=196`, `N=768`, `K=768`, `BM=32`, `BN=128` and
`BK=128`.  Each program computes a 32x128 logical tile; the launch grid is
7x6x1 (42 programs).  M=196 is not a multiple of BM=32, so the last M program
is the boundary tile. The frontend specializes the complete C load, K loop
and C store into `pid_m < 6` (32 valid rows) and an `else` (4 valid rows),
using the explicit 7x6x1 launch contract. The Triton source retains its masks
and six-iteration K loop.

The only lowering route in this directory is:

```text
Triton AST/TTIR -> external export_contract() -> verified standard Contract + manifest
     -> Buddy: contract verify -> bosc-ame-plan -> verified planned IR
     -> bosc-ame-emit -> lower-bosc-ame -> LLVM IR -> AME assembly -> ELF
```

## Quick start

`make ttir` exports without Buddy. For the complete image and simulator, set
`BUDDY_ROOT` to an external Buddy checkout, `BUDDY_MLIR_BINARY_DIR` to its
binary directory, and `LLVM_BIN` to the shared LLVM tool directory. Example:

```sh
export BUDDY_ROOT=/path/to/buddy-mlir
export BUDDY_MLIR_BINARY_DIR=/path/to/buddy-build/bin
export LLVM_BIN=/path/to/llvm-build/bin
```


```bash
make ttir            # external AOT -> TTIR, verified Contract and manifest
make boscame llvm    # TTIR -> frontend/bufferized/BOSCAME IR -> LLVM IR
make check           # typed ABI, AME instruction and NR ELF audits (no board)
make simcheck        # builds BOSCAME IR; full-grid and signed/wrap stress checks
make                 # full pipeline -> build/matmul_i8_196x768x768/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
The TTIR export and the generated ABI/config helpers import Triton, so those
Python steps run with the Triton/triton-riscv environment.  The makefile
uses explicit Buddy tool variables or PATH and the configured LLVM tools and accepts `PYTHON`, `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC`, `RISCV_CC` and linker overrides, for example:

```bash
make boscame llvm simcheck PYTHON=/path/to/boscame/python
```

## Tile contract

The AME register file holds at most 64 rows/columns per tile, so the requested
`BN=BK=128` is a *logical* tile: the planner lowers each program as two
64-column AME groups and each K block as two 64-wide AME K blocks.

| constant | value | why |
| --- | --- | --- |
| `M, N, K` | `196, 768, 768` | logical `C[M,N] = A[M,K] @ B[N,K]ᵀ` |
| `BM` | `32` | AME M tile; grid M = `ceil(196/32) = 7` |
| `BN` | `128` | logical N tile; two 64-column AME groups |
| `BK` | `128` | reduction block; two 64-wide AME K blocks |
| `GRID` | `7 × 6 × 1` | `ceil(M/BM) × N/BN` |

The kernel masks the rows past `M` in the final M program (and the K lanes past
`K` in the reduction loop) with `other=0`; the masked store never touches C
elements outside the logical 196 rows.  The accumulator is initialized from C
itself with `tl.load` and stored back once after the reduction. A compact
SPM C buffer carries the accumulator through all six K iterations. Each
iteration stages A through DMA, reads B directly from DDR, loads the C and
A/B subtiles into AME registers, accumulates directly into C, and writes the
accumulator back to the same C SPM window. Each K iteration
finishes its accumulator store before the next DMA; registers do not stay live
across runtime calls.
The launcher owns the logical
`A[196,768]`, `B[768,768]` and `C[196,768]` buffers and validates those rows
only.

All effective copy extents are static after specialization. Both branches
have full N/K extents; only the tail branch initializes invalid A/C rows.
B is physically K-contiguous in both branches. The frontend contract has 12
tasks and two matmuls; the planner folds the two B load snapshots into their
matmuls, leaving 10 tasks. The five modeled scopes are complete.

The planner keeps two physical SPM windows, totaling **20480 B**:

| Buffer | Logical shape | DDR strides (elements) | SPM strides | Bytes | Bank / byte offset |
| --- | --- | --- | --- | ---: | --- |
| A | `[32,128]` i8 | `[768,1]` | `[128,1]` | 4096 | 1 / 0 |
| B | `[128,128]` i8 | `[1,768]` | direct DDR | 0 | no window |
| C / running sum | `[32,128]` i32 | `[768,1]` | `[128,1]` | 16384 | 0 / 0 |

A is reused by both N subtiles, so DMA stages it once per K iteration. B has
identical external traffic through direct and staged routes; direct avoids
its 16 KiB window and DMA descriptors. Private C remains in SPM across the K
loop. There is no separate product P or RVV product addition. The mutually
exclusive full/tail arms share the two windows; four static `spm.view` ops
therefore describe two physical windows. Their bank high-water marks are
16384 B and 4096 B. Every DMA waits before computation or window reuse.
AME subtiles retain parent byte pitches: 128 for SPM A, 768 for DDR B and
512 for SPM C. The tail fills invalid A/C rows, moves only four valid rows,
and writes only those rows back to DDR. Each program reads and writes external
C once around all six K iterations.

The regenerated ELF's complete E6 call-chain stack bound is **1216 B**, leaving
64320 B in the unchanged 64 KiB stack. This includes boot, launcher, adapter,
MLIR wrapper, kernel spills, the AME encoder's temporary saves, DMA runtime,
both DMA failure hooks and the trap path. The corresponding reconstructed
baseline uses 2256 B. See the [validation record](../../../docs/BOSCAMEKernelContractValidation.md)
for hashes, assumptions and the reproducible linked-ELF audit.

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
triton-shared-opt build/matmul_i8_196x768x768/kernel.ttir \
  --triton-to-linalg-experimental="contract=true caller-contract=args_aligned64_nonalias launch-grid=triton_matmul_i8_196x768x768:7,6,1" \
  -o build/matmul_i8_196x768x768/kernel.frontend.mlir
buddy-opt build/matmul_i8_196x768x768/kernel.frontend.mlir \
  --kernel-contract-verify -o build/matmul_i8_196x768x768/kernel.bufferized.mlir
buddy-opt build/matmul_i8_196x768x768/kernel.bufferized.mlir \
  --bosc-ame-plan --bosc-ame-plan-verify \
  -o build/matmul_i8_196x768x768/kernel.planned.mlir
buddy-opt build/matmul_i8_196x768x768/kernel.planned.mlir \
  --bosc-ame-emit -o build/matmul_i8_196x768x768/kernel.boscame.mlir
buddy-opt build/matmul_i8_196x768x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/matmul_i8_196x768x768/kernel.llvm.mlir
buddy-translate build/matmul_i8_196x768x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/matmul_i8_196x768x768/kernel.ll
```

Contract mode performs integer normalization, static mask specialization and
safe bufferization inside the frontend. `kernel.frontend.mlir` already contains
standard buffer-form Linalg/MemRef/SCF and target-neutral `contract.*` facts;
`kernel.bufferized.mlir` is its verified copy, retained as a familiar artifact
name. No second bufferization is needed. The planner binds actual operands,
selects each DDR/SPM route, rebuilds content versions and dependencies, and
records concrete `plan.*` decisions in `kernel.planned.mlir`. Emission verifies
that plan and expands it mechanically. There is no global storage-policy option.

The contract can be produced by another frontend without Triton or BOSCAME
dialect dependencies. See the Buddy `BOSCAMEKernelContractPipeline.md` documentation
and Buddy `BOSCAMEKernelContractValidation.md` documentation
for the interface, resource counts and artifact hashes.

## ABI and image memory

The kernel C wrapper takes **three typed unranked memref descriptor pointers**
and six i32 grid/program scalars.  The unranked descriptor is
`{int64_t rank; void *descriptor;}` and points to a rank-one descriptor
`{allocated, aligned, offset, size, stride}`.  Descriptor offsets and sizes are
**elements**: C has `196*768` i32 elements, not bytes.

`build.py --adapter experimental-boscame` emits the real bridge to
`_mlir_ciface_triton_matmul_i8_196x768x768`.  It checks root alignment,
nonalias ranges, capacities and the expected physical strides, then executes
the full 7x6 grid (the boundary tile is launched, the kernel masks it).
`build.py --config` emits the C shape header used by the launcher, so a changed
Triton shape fails at compile time instead of reaching the adapter with stale
descriptors. `build.py --launch-grid` derives the symbol/grid contract from
the same constants; the frontend make recipe checks helper failure before
passing the contract to the compiler.

`link.ld` places the DDR operand windows at `0x80100000..0x802e0000` in PMP
entry 5, and the E6 stack at `0x802e0000..0x802f0000` in entry 6.  Entry 6
keeps the `0x0f` write-through policy during both DDR phases, so compiler
spills cannot switch between cached and uncached access across AME
operations.  The source window holds A (150528 B) and B (589824 B); the output
window holds C (602112 B).  Both fit their 1 MiB windows and the linker
asserts it.

The kernel assembly enables RVV and AME, passes through the AME word encoder
and instruction checks, and is linked into the NH/E6 image.  This directory
builds with `FENCES=0` by default: the assembly filter still checks every
encoded AME/RVV instruction but leaves out the board `fence rw, rw` pairs, so
`kernel.nr.s` shows the raw instruction stream and the ELF audit is relaxed
accordingly.  Build with `make FENCES=1` to restore the fenced image and the
strict fence audit.  Runtime/adapter C compilation keeps RVV disabled so
aggregate descriptor copies cannot introduce unaudited vector instructions.
`make check` checks the entry ABI, absence of heap calls, the presence of AME
MMA instructions, DDR/PMP layout and the full ELF audit.  These checks do not
execute a board.

## Validation

`make simcheck` runs every program of the 7x6 grid in the functional
`tools/boscame_buffer_sim.py` model with persistent poisoned SPM, checks the
DDR access ranges and guards, and compares all 196x768 = 150528 INT32 outputs
with an independent NumPy oracle. It then checks signed extrema, nonzero C
and INT32 wrap with tail-first and reverse launches, retaining polluted SPM
through both stress launches. The oracle computes in INT64 and converts to
INT32. Each program's DDR accesses must stay in its authorized rows/columns;
A/B and guards remain intact. All three launches compare exactly.

The compiler regression also covers unequal source/destination pitches,
nonzero offsets, non-square B, multiple partial program axes, rejected
overflow proofs, graph IDs/dependencies, compact capacities and final C
buffer reuse. Placement tests cover zero-trip/nested loops, in-place C updates,
exclusive branch reuse and rejection of escaping descriptors, calls and
unsupported regions. These changes preserve integer results bit for bit and do
not change floating reduction order.

```bash
make debug            # progress markers and a 20-row on-board oracle
make boardcheck       # two full launches with signed extremes, wrapping and guards
```

`make boardcheck` builds
`build/matmul_i8_196x768x768/boardcheck/matmul_i8_196x768x768.boardcheck.bin`.
The driver runs two different complete 42-program launches, with 301056 output
comparisons against its independent C oracle.  A/B include signed-i8 extremes,
B retains its asymmetric physical `[N,K]` layout, and nonzero C exercises i32
wrap.  The oracle uses unsigned arithmetic to define that wrap without
signed-overflow undefined behavior.  Inputs and before/after DDR guards are
checked after each round; the second launch reuses SPM without clearing it.
The shortened `debug` oracle does not replace this validation.

The full 196-row serial oracle is 115M multiply-adds per launch on the scalar
E6; keep the capture window in mind.  If the report shows `TIMEOUT` while
`checked_rows` was still growing and `errors=0`, the kernel finished and only
the check outran the platform's bounded wait -- use the `debug` image, whose
oracle covers the first 16 logical rows plus the final partial BM tile (rows
192..195), to get a verdict inside the window.  In both images the kernel
itself always computes the full result.

Upload the image through the usual platform command, then run it from the test
runner directory:

```bash
cp build/matmul_i8_196x768x768/matmul_i8_196x768x768.bin <tester>/test/
cd <tester> && ./run.sh test=matmul_i8_196x768x768.bin
```

The console shows the phase markers (`0x11` operands filled, `0x30` kernel
start, per-program `0x10xx` with `MATMUL_I8_TRACE`, `0x31` kernel returned,
`0x02`/`0xee` oracle verdict) and the final `stats phase=... cycles=...
checked_rows=... errors=... max_abs=...` report.  No board run has been
recorded for this migrated configuration yet.  The boot/platform setup is
shared with the other FPGA examples and is not part of the validation claim.

## Files

| File | Role |
| --- | --- |
| `kernels.py` | Triton computation with the reference masks and K loop; fixed shape/tile constants |
| `build.py` | Triton frontend export, launch-grid derivation, typed ABI adapter and C shape header, LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> AME assembly -> ELF/image pipeline and variants |
| `link.ld` | NH/E6 entries, DDR operand windows and stack cache policy |
| `launch.c` | Logical operands, board oracle and NH report |
| `simcheck.py` | Full-grid functional execution, numerical oracle and guard checks |
| `../tools/ame_to_word.py` | Fixed-register AME word encoder over the `llc` output |
| `../tools/restrict_fpga_assembly.py` | Restricted/fenced NR instruction stream |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |

See [the pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md).
