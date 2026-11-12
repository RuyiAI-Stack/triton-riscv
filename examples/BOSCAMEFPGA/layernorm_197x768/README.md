# Triton LayerNorm on the NH/E6 FPGA

The default build uses the **scalar bypass for ordered f32 reductions**:
sequential scalar accumulation in SSA, so the final ELF has no
`vfredosum.vs`. `make ALLOW_VECTOR_ORDERED_REDUCTIONS=1` restores the
original vector instruction. Switching back to `=0` does not need `clean`.
Range, stack cost and validation status are in
[the bypass note](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-ordered-reductions.md).
The 2026-10-28 production image passed two board rounds: 302592 outputs
matched bit for bit, errors and guards were 0, and cycles were 9851570 /
9847844. The task evidence is in that note. The current default planner
uses the direct DDR path. `make check` checks that path, or the older
four-DMA path, over the full external access range. Older performance and
stack records below belong to their own historical images.


The Triton frontend is provided by the external `triton-riscv` AOT exporter.
See [the Contract handoff guide](../../../docs/BoscameAOT.md) for installation,
tool selection, the API and migration of old commands.

This example lowers the ViT-B/16 LayerNorm operator over a `[197,768] f32`
hidden state through the experimental BOSCAME backend. The Triton source is
[kernel.py](kernel.py); the fixed problem size (`ROWS=197`, `WIDTH=768`,
`BLOCK=1024`, `EPS=1e-12`) and the one-program-per-row launch grid are
defined in [build.py](build.py). The operator has no dot: the reductions,
elementwise and select operations stay in Linalg and lower to SPM vector code.

The only lowering route in this directory is:

```text
Triton AST/TTIR -> external export_contract() -> verified standard Contract + manifest
     -> Buddy: contract verify -> bosc-ame-plan -> verified planned IR
     -> bosc-ame-emit -> lower-bosc-ame -> LLVM IR -> RVV assembly -> ELF
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
make adapter asm     # typed ABI bridge and the restricted NR assembly
make check           # typed ABI, boundary DMA and NR ELF audits (no board)
make simcheck        # functional SPM/DMA model over all 197 programs
make                 # full pipeline -> build/layernorm_197x768/*.bin
```

`make boscame`, `make llvm` and `make asm` stop at the named stage.
The TTIR export imports Triton, so that step runs with the Triton/triton-riscv
environment. The makefile uses explicit Buddy tool variables or PATH and the configured LLVM tools
and accepts `PYTHON`, `BUDDY_OPT`, `BUDDY_TRANSLATE`, `LLC`, `RISCV_CC` and linker
overrides, for example:

```bash
make boscame llvm simcheck PYTHON=/path/to/boscame/python
```

## Pipeline

The conversion commands used by `make boscame llvm` are:

```bash
triton-shared-opt build/layernorm_197x768/kernel.ttir \
  --triton-to-linalg-experimental="contract=true caller-contract=args_aligned64_nonalias launch-grid=triton_layernorm_197x768:197,1,1" \
  -o build/layernorm_197x768/kernel.frontend.mlir
buddy-opt build/layernorm_197x768/kernel.frontend.mlir \
  --kernel-contract-verify -o build/layernorm_197x768/kernel.bufferized.mlir
buddy-opt build/layernorm_197x768/kernel.bufferized.mlir \
  --bosc-ame-plan --bosc-ame-plan-verify \
  -o build/layernorm_197x768/kernel.planned.mlir
buddy-opt build/layernorm_197x768/kernel.planned.mlir \
  --bosc-ame-emit -o build/layernorm_197x768/kernel.boscame.mlir
buddy-opt build/layernorm_197x768/kernel.boscame.mlir \
  --lower-bosc-ame -o build/layernorm_197x768/kernel.llvm.mlir
buddy-translate build/layernorm_197x768/kernel.llvm.mlir \
  --buddy-to-llvmir -o build/layernorm_197x768/kernel.ll
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

## Memory and DMA

The planner places five 4 KiB float32 roots in SPM: a shared mean/invstd
broadcast buffer, X/centered/normalized/output, Weight, Bias and squares.
Bank 0 uses offsets 0, 4096 and 8192; bank 1 uses 0 and 4096. The bank
high-water marks are 12288 B and 8192 B, totaling **20480 B**. Safe frontend
bufferization preserves centered while squares is produced, then updates the
X/output root in place. No general lifetime-based physical window overlap is
assumed.

The logical tensor has 1024 lanes while only the first 768 are valid. Each
external DMA moves the 3072-byte prefix: X, Weight and Bias read, and Out
writes. Each load first fills all 1024 SPM lanes (X/Bias padding +0, Weight
padding 1). The output stays on SPM/DMA because the store covers only the valid
prefix of the private 1024-lane tile. Each program performs four DMA events,
9216 read bytes and 3072 write bytes; private arrays do not round-trip through
DDR.

| Buffer | DDR location | External DMA |
| --- | --- | --- |
| X | row × 3072 bytes | 3072-byte read |
| Weight | 3072 bytes | 3072-byte read |
| Bias | 3072 bytes | 3072-byte read |
| Out | row × 3072 bytes | 3072-byte write |

Both reductions use all 1024 logical lanes and divide by 768. The centered
prefix keeps the original subtraction; a subview fill writes exact +0 to
`[768,1024)` before squaring. Two ordered f32 reductions use 16-lane chunks
and `vfredosum.vs`, preserving the established reduction order and tolerance.
They write two private host rank-zero buffers, which emission converts to
aligned stack storage. Explicit scalar-load tasks capture the results before
the div/rsqrt chains. Mean and invstd are broadcast through the same SPM root
at different times; this version does not fuse those broadcasts into adjacent
RVV computations.

The plan has 21 tasks, versus 19 in the saved integrated baseline. The new
contract explicitly records two scalar-load events, two host initializations
and exact tail fill dependencies. Its five physical windows replace the
baseline's five windows with a 4 KiB gap: total bank high-water drops from
24576 B to 20480 B. The baseline's register broadcast forwarding is not part
of the current mechanical emitter, so the internal vector traffic differs.
External DMA ranges, floating-point operation order and output bytes remain
unchanged for the recorded full-grid validation.

## ABI and image memory

`build.py --adapter` emits the typed C bridge to
`_mlir_ciface_triton_layernorm_197x768`. The bridge takes **four typed
unranked f32 memref descriptor pointers** and six i32 grid/program scalars,
checks 64-byte root alignment, nonalias ranges, capacities and row-major
`[ROWS,WIDTH]` / `[WIDTH]` strides, then launches all 197 row programs
in reverse order with progress markers.

`link.ld` places user DDR at `0x80100000..0x802e0000` in PMP entry 5, and
the E6 stack at `0x802e0000..0x802f0000` in entry 6. Entry 6 keeps the
`0x0f` write-through policy during both DDR phases, so descriptor or stack
spills cannot be read through a stale cached entry after a policy switch. The
freestanding image links the NH/E6 boot/platform code, the DMA/cache-policy
runtime and `nr_math.c` for `sqrtf`/`rsqrtf`; `-ffp-contract=off` keeps the
square root and division separately rounded.

## Validation

The 2026-10-28 production SSA scalar image passed on FPGA4: two input sets,
394 row programs, all 302592 outputs matched bit for bit, `errors=0` and
`guard_errors=0`. The two rounds were 9851570 / 9847844 cycles. The task id
is `661e98c04c27457ba59273b9adb8ce83`. The image SHA256 is
`58fd9d39faa5124c8b7348a79345144460899651ea7c8233ae4f953dd05de04c`. The full
task, UART and diagnosis history are in
[the validation report](../../../buddy-mlir/reference/rvv_reduction_probe/scalar-workaround/README.md).

`make check` checks the typed LLVM ABI, the absence of heap allocation,
full 197-row access coverage of the real IR, and the complete ELF. The
current image has 1911 static instructions, 29 RVV, 0 AME and 0
`vfredosum`. The reduction seed and result do not bounce through `vfmv`.
The call-chain stack audit is 1056/65536 bytes.

`make simcheck` runs all 197 rows in reverse order against a sequential f32
oracle, with `atol=rtol=2e-5`. The current direct-DDR path covers X three
times, Weight once, Bias once and Out once. Guards are intact and the
maximum absolute error is 0. The older four-DMA path is still checked
against its own contract. `--write-output FILE` saves the 605184-byte
output. `--compare-output FILE` compares it byte for byte with an existing
output. Functional simulation and the instruction audit do not replace the
hardware numerical acceptance above.

Run from the repository root:

```sh
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
export BUDDY_MLIR_BINARY_DIR="$BUDDY_ROOT/build/bin"
export LLVM_BIN="$BUDDY_ROOT/llvm/build/bin"
make -C examples/BOSCAMEFPGA/layernorm_197x768 check simcheck
python examples/BOSCAMEFPGA/run_bin.py examples/BOSCAMEFPGA/layernorm_197x768/build/layernorm_197x768/layernorm_197x768.bin
```

Reuse an existing SSH tunnel, run through the existing scheduler queue, and
keep the original client until the task ends. 120 seconds is the execution
limit. Queue time is separate. Record the actual task, node, FPGA and image
SHA256. This image is fully checked by the board oracle and does not declare
DDR output. Do not treat a load readback as the run result. The older
standalone DDR-dump checker can still be used on historical data. It is not
the source of this acceptance. The service did not provide a measured clock,
so cycles are not converted to time from the UART frequency label.

## Files

| File | Role |
| --- | --- |
| `kernel.py` | Original Triton computation |
| `build.py` | Triton frontend export, fixed shape, typed ABI adapter and LLVM ABI check |
| `makefile` | TTIR -> BOSCAME -> LLVM -> assembly -> ELF/image pipeline |
| `link.ld` | NH/E6 entries, user DDR windows and stack cache policy |
| `launch.c` | Operands, two-round guarded f32 board oracle and NH report |
| `simcheck.py` | Whole-program functional execution, numerical oracle and guard checks |
| `../tools/restrict_fpga_assembly.py` | Restricted NR instruction stream from the `llc` assembly |
| `../tools/check_nr_elf.py` | Complete-ELF instruction audit |

See the Buddy `BOSCAMEKernelContractPipeline.md` documentation
and [current validation record](../../../docs/BOSCAMEKernelContractValidation.md).
