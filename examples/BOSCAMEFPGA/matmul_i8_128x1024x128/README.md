# Triton INT8 GEMM on the NH/E6 FPGA

The Triton frontend is provided by the external `triton-riscv` AOT exporter.
See [the Contract handoff guide](../../../docs/BoscameAOT.md) for installation,
tool selection, the API and migration of old commands.

This example lowers `A[128,128] i8 @ B[1024,128] i8ᵀ + C[128,1024] i32`
through the experimental BOSCAME backend. B is physically `[N,K]`, with K
contiguous. The Triton source is [kernels.py](kernels.py); the global
compile-time constants are `M=128`, `N=1024`, `K=128`, `BM=64`, `BN=64` and
`BK=128`. Each program computes a 64x64 tile; the launch grid is 2x16x1
(32 programs). The single Triton K block is lowered to two internal 64-wide
AME blocks.

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
make simcheck        # functional DMA/SPM/AME model over the whole grid
make                 # full pipeline -> build/matmul_i8_128x1024x128/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
The TTIR export and the generated ABI/config helpers import Triton, so those
Python steps run with the Triton/triton-riscv environment. The makefile
uses explicit Buddy tool variables or PATH and the configured LLVM tools and accepts `PYTHON`, `BUDDY_OPT`,
`BUDDY_TRANSLATE`, `LLC`, `RISCV_CC` and linker overrides, for example:

```bash
make boscame llvm simcheck PYTHON=/path/to/boscame/python
```

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
triton-shared-opt build/matmul_i8_128x1024x128/kernel.ttir \
  --triton-to-linalg-experimental="contract=true caller-contract=args_aligned64_nonalias launch-grid=triton_matmul_i8_128x1024x128:2,16,1" \
  -o build/matmul_i8_128x1024x128/kernel.frontend.mlir
buddy-opt build/matmul_i8_128x1024x128/kernel.frontend.mlir \
  --kernel-contract-verify -o build/matmul_i8_128x1024x128/kernel.bufferized.mlir
buddy-opt build/matmul_i8_128x1024x128/kernel.bufferized.mlir \
  --bosc-ame-plan --bosc-ame-plan-verify \
  -o build/matmul_i8_128x1024x128/kernel.planned.mlir
buddy-opt build/matmul_i8_128x1024x128/kernel.planned.mlir \
  --bosc-ame-emit -o build/matmul_i8_128x1024x128/kernel.boscame.mlir
buddy-opt build/matmul_i8_128x1024x128/kernel.boscame.mlir \
  --lower-bosc-ame -o build/matmul_i8_128x1024x128/kernel.llvm.mlir
buddy-translate build/matmul_i8_128x1024x128/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/matmul_i8_128x1024x128/kernel.ll
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

## AME datapath

The signed integer `dot(A,B,C)` becomes a tensor `linalg.matmul` with C as
its initial accumulator before bufferization. The storage planner proves that
the complete private C snapshot and terminal writeback can forward to the
external C window. No separate product buffer or RVV product addition remains.

For this shape and layout the planner proves that A, B and previous C are
valid direct-DDR AME operands: their physical strides and the aligned/nonalias
DMA ABI match the required tile layouts. The generated image consumes them
directly with `bosc_ame.tile.load_a`, `tile.load_b`, `tile.load_acc`,
`tile.mma` and `tile.store_acc` carrying `bosc_ame.direct_ddr`, so this
kernel needs no program-local DMA or SPM window. K=128 is split into two
64-wide AME blocks; C is read and written once around them.

## ABI and image memory

The kernel C wrapper takes **three typed unranked memref descriptor pointers**
and six i32 grid/program scalars. The unranked descriptor is
`{int64_t rank; void *descriptor;}` and points to a rank-one descriptor
`{allocated, aligned, offset, size, stride}`. Descriptor offsets and sizes are
**elements**: C has 131072 i32 elements, not 524288 i8 elements.

`build.py --adapter experimental-boscame` emits the real bridge to
`_mlir_ciface_triton_matmul_i8_128x1024x128`. It checks root alignment,
nonalias ranges, capacities and the expected physical strides, then executes
the full grid. `build.py --config` emits the C shape header used by the
launcher, so a changed Triton shape fails at compile time instead of reaching
the adapter with stale descriptors.

`link.ld` places the DDR operand windows at `0x80100000..0x802e0000` in PMP
entry 5, and the E6 stack at `0x802e0000..0x802f0000` in entry 6. Entry 6
keeps the `0x0f` write-through policy during both DDR phases, so compiler
spills cannot switch between cached and uncached access across AME
operations.

The kernel assembly enables RVV and AME, passes through the AME word encoder
and instruction checks, and is linked into the NH/E6 image. This directory
builds with `FENCES=0` by default: the assembly filter still checks every
encoded AME/RVV instruction but leaves out the board `fence rw, rw` pairs, so
`kernel.nr.s` shows the raw instruction stream and the ELF audit is relaxed
accordingly. Build with `make FENCES=1` to restore the fenced image and the
strict fence audit. Runtime/adapter C compilation keeps RVV disabled so
aggregate descriptor copies cannot introduce unaudited vector instructions.
`make check` checks the entry ABI, absence of heap calls, expected AME
instructions, RVV memory instructions, DDR/PMP layout and the full ELF
allowlist. These checks do not execute a board.

## Validation

`make simcheck` runs every program of the grid in the functional
`tools/boscame_buffer_sim.py` model with persistent poisoned SPM, checks DDR
access ranges and guards, and compares all 131072 results with an independent
NumPy INT32 oracle. The compiler regression suite additionally uses signed
extrema, random nonzero C and i32 overflow; a separate K=256 test checks
internal blocking, a nonzero dot initializer and rectangular masks. All integer
comparisons are exact.

The regenerated contract/plan/emit build passes `make`, `make check` and
`make simcheck`: all 32 programs produce 131072 exact i32 outputs, with zero
DMA transfers and zero SPM occupancy. Five source tasks fold into one matmul
task. Its final kernel assembly matches the saved baseline byte for byte.
The 524288 verified output bytes have SHA256
`38114b2648726d06482fc92bf727d2376ae93b763c1951c7df6cd16876e45cb2`.
The linked ELF's E6 call-chain stack bound is 656 B, including the encoder's
temporary saves, within the unchanged 65536-byte stack. DMA code is unreachable
and removed from this direct-only image. See the [validation record](../../../docs/BOSCAMEKernelContractValidation.md)
for the current artifact hashes and audit procedure.

```bash
make debug            # progress markers and the launcher's 16-row board oracle
make boardcheck       # two full launches with signed extremes, wrapping and guards
```

`make boardcheck` builds
`build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.bin`.
The driver runs two different complete 32-program launches, with 262144 output
comparisons against its independent C oracle. A/B include signed-i8 extremes,
B retains its asymmetric physical `[N,K]` layout, and nonzero C exercises i32
wrap. The oracle uses unsigned arithmetic to define that wrap without
signed-overflow undefined behavior. Inputs and before/after DDR guards are
checked after each round; the second launch reuses SPM without clearing it.
The shortened `debug` oracle does not replace this validation.

Upload this image using the usual platform command, then run it from the test
runner directory:

```bash
./run.sh test=test/matmul_i8_128x1024x128.boardcheck.bin
```

Keep the session alive until the terminal application verdict. Read back DDR
after that verdict, starting at `0x80000000` through the outputs and
diagnostics; a 3 MiB dump covers this image. The loader's pre-execution mirror
cannot be used as output evidence. From this example directory, validate the
final dump using the exact ELF used to build the uploaded image:

```bash
python3 ../tools/check_experimental_board.py gemm \
  --elf build/matmul_i8_128x1024x128/boardcheck/matmul_i8_128x1024x128.boardcheck.elf \
  --dump gemm.final-ddr.bin --report gemm.board.json
```

The checker independently computes the full final-round product in i64, adds
initial C and explicitly wraps to i32. Every output bit must match. It also
verifies both rounds' counters and the final input bytes/DDR guards; missing
data or any mismatch returns nonzero. The JSON report includes ELF/dump hashes
and comparison counts. Stored board reports in
[docs/validation](../../../docs/validation/) cover earlier images and do not
validate the current 64x64-tile configuration. No controlled baseline latency
comparison has been made.

## Files

| File | Role |
| --- | --- |
| `kernels.py` | Original Triton computation and fixed shape/tile constants |
| `build.py` | Triton frontend export, fixed shape, typed ABI adapter and C shape header, LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> AME assembly -> ELF/image pipeline and variants |
| `link.ld` | NH/E6 entries, DDR operand windows and stack cache policy |
| `launch.c` | Operands, two-round guarded INT32 board oracle and NH report |
| `simcheck.py` | Full-grid functional execution, numerical oracle and guard checks |
| `../tools/ame_to_word.py` | Fixed-register AME word encoder over the `llc` output |
| `../tools/restrict_fpga_assembly.py` | Restricted/fenced NR instruction stream |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |
| `../tools/check_experimental_board.py` | Independent final-DDR checker using exact ELF symbols |

See [the pipeline guide](../../../docs/BOSCAMEKernelContractPipeline.md).
