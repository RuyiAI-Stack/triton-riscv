# RISC-V ELF Generation and QEMU Execution

This guide compiles a Python `@triton.jit` kernel into a standalone RISC-V
Linux executable, runs it with QEMU user-mode emulation, and verifies its
outputs. The helper generates the C runner automatically; you provide only the
kernel, arguments, launch grid, and expected values.

> **Best for:** readers who have completed the Host CPU Quick Start and now
> want to prove that one kernel can run as RISC-V code.
>
> **Prerequisites:** a source build plus a RISC-V GNU toolchain, sysroot, and
> QEMU; see [Build from Source](00-Getting-Started.md#build-from-source).
>
> **After this guide:** you can generate a RISC-V ELF, execute it with QEMU,
> verify its result, and distinguish emulated execution from Host CPU execution.

## Minimal RISC-V/QEMU Tutorial

This is the shortest path through this page. It uses the repository's existing
`python/examples/rvv_vec_add_elf.py`; you do not need to write a runner or a new
kernel.

Unlike the first example in [Getting started](00-Getting-Started.md), this path
really does cross-compile RISC-V machine code and execute it under QEMU. QEMU is
an emulator, so a successful run proves correctness for the embedded test case,
not performance on physical RISC-V hardware.

### Step 1: Load the Source-Build Environment

From the Triton-RISCV repository:

```sh
cd /path/to/workspace/triton-riscv
source .venv/bin/activate
source scripts/triton-riscv-env.sh
```

If you have not built the toolchain yet, stop here and complete
[Build the GNU Toolchain and QEMU](#build-the-gnu-toolchain-and-qemu).

### Step 2: Check the Compiler, Sysroot, and QEMU

```sh
(
  set -euo pipefail
  printf 'CC=%s\nSYSROOT=%s\nQEMU=%s\n' \
    "$TRITON_RISCV_CC" "$TRITON_RISCV_SYSROOT" "$TRITON_RISCV_QEMU"

  test -x "$TRITON_RISCV_CC"
  test -x "$TRITON_RISCV_QEMU"
  test -d "$TRITON_RISCV_SYSROOT"
  "$TRITON_RISCV_CC" --version | head -n1
  "$TRITON_RISCV_QEMU" --version | head -n1
  echo 'PASS: RISC-V compiler, sysroot, and QEMU are available'
)
```

Success means that both version commands print a version, the `test` command
does not stop the shell, and the final `PASS` line appears. Empty variables or
`No such file or directory` mean the target environment is not ready.

### Step 3: Install the Current Plugin Code

```sh
scripts/rebuild-triton-riscv.sh
```

The command succeeds when it exits with status 0 and the final package build or
installation command reports no error. Re-run it whenever you change backend
Python or C++ code.

### Step 4: Generate the ELF and Run It with QEMU

```sh
python python/examples/rvv_vec_add_elf.py
```

A successful run ends with output similar to:

```text
output_ptr=[0,3,6,...,189]
PASS
ELF: artifacts/riscv/rvv-vector-add.elf
```

The exact float formatting may differ. The important signals are `PASS`, a
nonzero-sized ELF, and no QEMU verification error:

```sh
(
  set -e
  test -s artifacts/riscv/rvv-vector-add.elf
  echo 'PASS: RISC-V ELF was generated'
)
```

The script's `expected={...}` values are compiled into the generated runner.
The runner checks `output_ptr` after all program instances execute, so `PASS`
means more than “QEMU did not crash”: the vector-add results matched too.

## How This Workflow Works

```text
Python @triton.jit kernel
  -> TTIR / structured MLIR / Vector / LLVM IR
  -> RV64 object
  -> generated C runner
  -> static link with a RISC-V GNU toolchain
  -> ELF64 RISC-V executable
  -> QEMU execution and output checks
```

A passing run proves that the selected kernel can be lowered for the configured
RISC-V target, linked with the target ABI, executed by QEMU, and produce the
expected values for the embedded test data.

It does not prove performance on real hardware, full compatibility with every
RISC-V core, or support for arbitrary runtime-shaped inputs.

## Prerequisites

Complete [Build from Source](00-Getting-Started.md#build-from-source),
including these sibling checkouts:

```text
workspace/
  triton-riscv/
  triton/
  buddy-mlir/
```

You need:

- a working Triton-RISCV source build;
- Buddy's LLVM/MLIR build under `buddy-mlir/llvm/build` or equivalent
  `LLVM_DIR`/`MLIR_DIR` overrides;
- enough disk space and time to build a RISC-V GNU toolchain, glibc, and QEMU;
- Bash, CMake, Ninja, a host C/C++ toolchain, and Python.

The fast prebuilt setup alone does not contain the RISC-V GNU sysroot and QEMU
used by this workflow. You may reuse an existing compatible cross-toolchain,
sysroot, and QEMU by setting the environment variables in
[Use an Existing Toolchain](#use-an-existing-toolchain).

## Build the GNU Toolchain and QEMU

From the Triton-RISCV repository:

```sh
cd /path/to/workspace/triton-riscv
source .venv/bin/activate
scripts/build-riscv-gnu-toolchain.sh
```

The script configures a dedicated Buddy build with:

- `BUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN=ON`;
- `BUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON`;
- the LLVM and MLIR CMake packages from the Buddy checkout.

It then runs the full Ninja build. The default output is:

```text
buddy-mlir/build-for-triton-riscv/thirdparty/riscv-gnu-toolchain/
  bin/riscv64-unknown-linux-gnu-gcc
  bin/riscv64-unknown-linux-gnu-objdump
  bin/qemu-riscv64
  sysroot/
```

Override the source, build directory, or parallelism when needed:

```sh
export BUDDY_DIR=/path/to/buddy-mlir
export BUDDY_BUILD_DIR=/path/to/buddy-riscv-build
export JOBS=8
scripts/build-riscv-gnu-toolchain.sh
```

The environment helper sets `PYTHONSAFEPATH=1`; the build script temporarily
removes it while building glibc so glibc's Python generators can import local
modules.

## Load and Verify the Target Environment

Re-source the environment so it detects the newly built tools:

```sh
cd /path/to/workspace/triton-riscv
source .venv/bin/activate
source scripts/triton-riscv-env.sh
```

Inspect every target path before compiling:

```sh
printf 'CC=%s\n' "$TRITON_RISCV_CC"
printf 'OBJDUMP=%s\n' "$TRITON_RISCV_OBJDUMP"
printf 'SYSROOT=%s\n' "$TRITON_RISCV_SYSROOT"
printf 'QEMU=%s\n' "$TRITON_RISCV_QEMU"

"$TRITON_RISCV_CC" --version | head -n1
"$TRITON_RISCV_QEMU" --version | head -n1
```

The default target settings from `backend/riscv.py` are:

| Setting | Default |
| --- | --- |
| target triple | `riscv64-unknown-linux-gnu` |
| architecture | `rv64gcv_zfh_zvfh_zba_zbb` |
| ABI | `lp64d` |
| LLVM features | `+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb` |

These settings require a compatible compiler, sysroot, QEMU CPU, and real
hardware. Override them together rather than changing only the linker `-march`
or only the LLVM feature set.

Rebuild the plugin so the installed package contains the current RISC-V helper:

```sh
scripts/rebuild-triton-riscv.sh
```

## Example CLI Options

The minimal tutorial used the default compile-and-run mode:

```sh
python python/examples/rvv_vec_add_elf.py
```

The script:

1. specializes `vector_add_kernel` for `fp32` buffers and `BLOCK_SIZE=16`;
2. cross-compiles it to a RISC-V object;
3. embeds 64 input values and an output buffer in a generated runner;
4. links `artifacts/riscv/rvv-vector-add.elf` statically;
5. runs it with QEMU;
6. compares every output value with the expected sum.

A successful run ends with output similar to:

```text
output_ptr=[0,3,6,...,189]
PASS
ELF: artifacts/riscv/rvv-vector-add.elf
```

Compile without running QEMU:

```sh
python python/examples/rvv_vec_add_elf.py --compile-only
```

Compile and write disassembly beside the ELF:

```sh
python python/examples/rvv_vec_add_elf.py --compile-only --dump-asm
```

Choose different output paths:

```sh
python python/examples/rvv_vec_add_elf.py \
  --compile-only \
  --output artifacts/riscv/add.elf \
  --dump-asm \
  --asm-output artifacts/riscv/add.s
```

Run an existing ELF separately:

```sh
python -m triton.backends.triton_shared.riscv run \
  artifacts/riscv/rvv-vector-add.elf
```

## Compile Your Own Kernel

The remaining sections are advanced reference for kernel developers and
compiler contributors. Finish the minimal tutorial before replacing its kernel.

`standalone_kernel_cli` gives a script the same `--output`, `--compile-only`,
`--dump-asm`, and `--asm-output` options as the example:

```python
import triton
import triton.language as tl
from triton.backends.triton_shared.riscv import standalone_kernel_cli


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * SCALE, mask=mask)


def main():
    size = 33
    block = 16
    scale = 2.0
    values = [float(i) for i in range(size)]

    standalone_kernel_cli(
        scale_kernel,
        arguments={
            "x_ptr": values,
            "out_ptr": [0.0] * size,
            "n": size,
        },
        constexprs={"SCALE": scale, "BLOCK": block},
        grid=(triton.cdiv(size, block),),
        expected={"out_ptr": [value * scale for value in values]},
        default_output="artifacts/riscv/scale.elf",
    )


if __name__ == "__main__":
    main()
```

For integration without command-line argument parsing, use
`compile_kernel_to_elf` or `compile_and_run_kernel` with the same kernel data.

### Argument rules

`arguments` contains every non-`constexpr` kernel argument by name. Common
types are inferred from Python or array-like values:

- float buffers become `*fp32`;
- integer buffers become `*i32` or the dtype reported by the array object;
- scalar floats become `fp32`;
- scalar integers become `i32` when in range, otherwise `i64`.

Use `signature={...}` for unsigned integers, `fp64`, empty buffers, or any case
where inference is ambiguous. The helper currently supports pointer and scalar
forms of signed/unsigned 8-, 16-, 32-, and 64-bit integers, plus `fp32` and
`fp64`. Automatic `fp16`/`bf16` serialization is not implemented.

### Grid rules

`grid` may have one, two, or three dimensions. The generated runner normalizes
missing dimensions to one and invokes the kernel for every `(x, y, z)` program
ID. Make sure the embedded buffers are large enough for every program instance.

### Expected-output rules

`expected` maps output argument names to the values that should exist after all
program instances run. A mismatch prints a verification error and exits
nonzero. Use `atol=` for floating-point cases that need a justified absolute
tolerance.

## Inspect the ELF and Assembly

The example's `--dump-asm` option is the simplest route. For independent
inspection:

```sh
"$LLVM_BINARY_DIR/llvm-readelf" \
  -h artifacts/riscv/rvv-vector-add.elf

"$LLVM_BINARY_DIR/llvm-objdump" \
  -d --mattr=+v artifacts/riscv/rvv-vector-add.elf \
  | rg 'vset|vle|vse|vfadd'
```

The ELF header should report:

- `Class: ELF64`;
- `Type: EXEC`;
- `Machine: RISC-V`.

For vector addition, look for RVV setup, load, arithmetic, and store
instructions such as `vsetvli`/`vsetivli`, `vle32.v`, `vfadd.vv`, and
`vse32.v`. Exact scheduling and register choices can change with LLVM.

Also check for unwanted behavior:

```sh
rg -n 'malloc|free' artifacts/riscv/rvv-vector-add.s
```

An unexpected allocation or copy is an optimization clue. It is not safe to
remove until aliasing, masks, bounds, strides, and lifetimes are proven.

## Run on Real RISC-V Linux

The generated executable is statically linked, so you can copy it to a machine
whose ISA supports the configured `-march`:

```sh
scp artifacts/riscv/rvv-vector-add.elf user@riscv-host:/tmp/
ssh user@riscv-host /tmp/rvv-vector-add.elf
```

`PASS` on hardware validates the embedded case. Benchmark only after confirming
the CPU's supported ISA, vector length behavior, clock/power conditions, and a
measurement method that separates startup from kernel execution.

## Use an Existing Toolchain

Set these variables before sourcing the environment helper, or export them in
the shell used to run the example:

```sh
export TRITON_RISCV_CC=/opt/riscv/bin/riscv64-linux-gnu-gcc
export TRITON_RISCV_OBJDUMP=/opt/riscv/bin/riscv64-linux-gnu-objdump
export TRITON_RISCV_SYSROOT=/opt/riscv/sysroot
export TRITON_RISCV_QEMU=/usr/bin/qemu-riscv64

export TRITON_RISCV_TARGET_TRIPLE=riscv64-unknown-linux-gnu
export TRITON_RISCV_MARCH=rv64gcv_zfh_zvfh_zba_zbb
export TRITON_RISCV_MABI=lp64d
export TRITON_RISCV_LLC_FEATURES=+m,+a,+f,+d,+c,+v,+zfh,+zvfh,+zba,+zbb
```

The compiler, linker, sysroot, QEMU, and destination hardware must agree on the
ABI and required extensions.

## Troubleshooting

### `Unable to find RISC-V C compiler` or QEMU

Re-source the environment after building the toolchain:

```sh
source scripts/triton-riscv-env.sh
```

If the toolchain is outside the default Buddy build, set
`RISCV_GNU_TOOLCHAIN_DIR` before sourcing, or set the individual variables from
[Use an Existing Toolchain](#use-an-existing-toolchain).

### Backend changes are not reflected

```sh
scripts/rebuild-triton-riscv.sh
export TRITON_CACHE_DIR="$PWD/artifacts/cache/rvv-debug"
```

Then rerun the example with a fresh cache.

### Linker reports soft-float/double-float incompatibility

Confirm that LLVM features include `+f,+d`, the linker uses `-mabi=lp64d`, and
the sysroot libraries were built for the same ABI. Do not work around the error
by mixing incompatible libraries.

### QEMU reports an illegal instruction

The generated `-march`/LLVM features require an extension not enabled by the
selected QEMU CPU. Inspect the assembly and align
`TRITON_RISCV_MARCH`, `TRITON_RISCV_LLC_FEATURES`, and
`TRITON_RISCV_QEMU_CPU` with the QEMU build. Lowering the advertised ISA is
valid only if the kernel is regenerated with the same reduced feature set.

### QEMU reports `verification failed`

The ELF executed, but at least one output differed. Recheck argument order and
types, grid dimensions, `tl.constexpr` values, buffer lengths, mask behavior,
and expected output. Use a smaller input and inspect IR/assembly around the
first differing element.

### No RVV instructions appear

First confirm the ELF was compiled with vector features. Then inspect
`vector.mlir`: the operation may not have vectorized, may have scalarized, or
may be obscured by temporary buffers/non-contiguous access. See
[Optimization opportunities](07-Optimization.md).

[Previous: Operator migration](05-Operator-Migration.md) · [Documentation index](README.md) · [Next: Optimization opportunities](07-Optimization.md)
