# End-to-End RVV ELF Generation and QEMU Execution

This document explains how to compile a Python Triton kernel directly into a RISC-V Linux ELF executable and run it with QEMU user-mode emulation. The workflow automatically generates the runner, compiles an RVV object, links the ELF, launches QEMU, and verifies the results. Users do not need to write or maintain a C/C++ runner.

## 1. End-to-End Pipeline

```text
Python @triton.jit kernel
  -> TTIR / Linalg / Vector / LLVM IR
  -> RV64 RVV object
  -> automatically generated temporary runner
  -> static link with riscv64-unknown-linux-gnu-gcc
  -> ELF64 RISC-V executable
  -> QEMU execution and output verification
```

## 2. Build the GNU Toolchain and QEMU from Buddy Compiler

The default repository layout is:

```text
triton-riscv/
buddy-mlir/
triton/
```

`triton-riscv` provides a build script that enables buddy-mlir
`BUDDY_MLIR_ENABLE_RISCV_GNU_TOOLCHAIN` CMake option and runs the full Buddy
Ninja build in a dedicated build directory. It also enables Buddy's Python
packages. The script does not run any Git submodule initialization commands;
toolchain configuration and building are invoked only through Buddy's CMake.

```sh
conda activate ruyiai
cd /path/to/triton-riscv
scripts/build-riscv-gnu-toolchain.sh
```

The default installation directory is:

```text
../buddy-mlir/build-for-triton-riscv/thirdparty/riscv-gnu-toolchain/
  bin/riscv64-unknown-linux-gnu-gcc
  bin/qemu-riscv64
  sysroot/
```

Use `BUDDY_DIR`, `BUDDY_BUILD_DIR`, and `JOBS` to override the Buddy source
directory, Buddy build directory, and build parallelism. The toolchain is
installed under `${BUDDY_BUILD_DIR}/thirdparty/riscv-gnu-toolchain` by Buddy's
CMake configuration.

> `scripts/triton-riscv-env.sh` sets `PYTHONSAFEPATH=1`. The toolchain build script automatically removes this variable while building glibc. Otherwise, glibc's generator scripts cannot import Python modules from their own directory.

## 3. Rebuild the Triton-RISCV Backend

The ELF APIs are part of the Triton backend plugin and must be synchronized into the `ruyiai` environment:

```sh
conda activate ruyiai
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
scripts/rebuild-triton-riscv.sh
```

The environment script automatically detects GCC, the sysroot, and QEMU under the Buddy build directory. No toolchain paths need to be configured manually. To inspect the detected paths:

```sh
printf '%s\n' "$TRITON_RISCV_CC"
printf '%s\n' "$TRITON_RISCV_SYSROOT"
printf '%s\n' "$TRITON_RISCV_QEMU"
```

## 4. Generate and Run an ELF with One Command

The repository includes a complete vector-add example:

```sh
conda activate ruyiai
cd /path/to/triton-riscv
source scripts/triton-riscv-env.sh
python python/examples/rvv_vec_add_elf.py
```

A successful run prints the inputs, output, verification result, and ELF path:

```text
output_ptr=[0,3,6,...,189]
PASS
ELF: artifacts/riscv/rvv-vector-add.elf
```

To generate the ELF without executing it:

```sh
python python/examples/rvv_vec_add_elf.py --compile-only
```

To also dump the final ELF disassembly:

```sh
python python/examples/rvv_vec_add_elf.py --compile-only --dump-asm
```

This writes `artifacts/riscv/rvv-vector-add.s` by default. Use `--asm-output PATH` to select another path.

The generated ELF can then be executed independently:

```sh
python -m triton.backends.triton_shared.riscv run \
  artifacts/riscv/rvv-vector-add.elf
```

## 5. Compile a Custom Triton Kernel

Use `standalone_kernel_cli` when a kernel should expose the same `--compile-only`, `--dump-asm`, `--asm-output`, and `--output` options as the vector-add example. The helper owns all CLI, ASTSource, runner, linking, and QEMU logic. A kernel only supplies its test data and launch configuration:

```python
from triton.backends.triton_shared.riscv import standalone_kernel_cli

standalone_kernel_cli(
    my_kernel,
    arguments={
        "x_ptr": input_values,
        "output_ptr": [0.0] * len(input_values),
        "n_elements": len(input_values),
    },
    constexprs={"BLOCK_SIZE": 16},
    grid=(triton.cdiv(len(input_values), 16),),
    expected={"output_ptr": expected_values},
    default_output="artifacts/riscv/my-kernel.elf",
)
```

Common signatures are inferred from the values: Python float buffers become `*fp32`, integer buffers become `*i32`, and scalar floats and integers become `fp32` and `i32`. Pass `signature={...}` only for ambiguous or non-default types such as unsigned integers or `fp64`. Empty buffers also require an explicit signature.

For integration into another Python application without a command-line interface, use `compile_kernel_to_elf` or `compile_and_run_kernel` with the same arguments.

Python buffers are embedded into the automatically generated temporary runner. The optional `expected` mapping enables output verification inside QEMU; a mismatch produces a nonzero exit status. The data, grid, constexprs, and expected output are kernel-specific and are the only parts each kernel must provide.

The automatic runner currently supports:

- Pointer arguments: `*i8`, `*i16`, `*i32`, `*i64`, the corresponding unsigned types, `*fp32`, and `*fp64`.
- Scalar arguments with the same integer and floating-point types.
- One-, two-, and three-dimensional launch grids.
- Static Linux ELF executables and QEMU user-mode execution.

Automatic serialization for `fp16`, `bf16`, dynamic input files, and complex memref shapes is not yet implemented.

## 6. Inspect the ELF and RVV Instructions

```sh
../buddy-mlir/llvm/build/bin/llvm-readelf \
  -h artifacts/riscv/rvv-vector-add.elf

../buddy-mlir/llvm/build/bin/llvm-objdump \
  -d --mattr=+v artifacts/riscv/rvv-vector-add.elf \
  | rg 'vset|vle|vse|vfadd'
```

Expected ELF properties include:

- `Class: ELF64`
- `Type: EXEC`
- `Machine: RISC-V`

The disassembly should contain RVV instructions such as `vsetvli` or `vsetivli`, `vle32.v`, `vfadd.vv`, and `vse32.v`.

## 7. Run FlagGems Operators with x86 and QEMU

The same QEMU route can be used for the FlagGems operator examples. The
environment may be a virtualenv or a conda environment; `scripts/triton-riscv-env.sh`
uses `CONDA_PREFIX` as `TRITON_VENV` when it is active.

```sh
conda activate /path/to/triton-riscv/.conda-env
cd /path/to/triton-riscv
TRITON_VENV="${CONDA_PREFIX}" source scripts/triton-riscv-env.sh
scripts/rebuild-triton-riscv.sh
```

Then run the issue #35 validation command:

```sh
pytest python/examples/ \
  --ignore=python/examples/test_core.py \
  --ignore=python/examples/test_annotations.py \
  -v
```

The current validated result is `3892 passed, 280 warnings`, with no FlagGems
file-level or parameter-level skips. `python/examples/conftest.py` retains only
global guards for unsupported parameter classes such as bfloat16, TF32, and
float8; none are exercised by this baseline.

See [FLAGGEMS-SKIP-REASONS.md](FLAGGEMS-SKIP-REASONS.md) for the validation
status and the backend coverage that replaced the former skip list.

## 8. Troubleshooting

### GCC or QEMU Is Not Found

Source the environment script again:

```sh
source scripts/triton-riscv-env.sh
```

For a non-default installation directory, set `RISCV_GNU_TOOLCHAIN_DIR` before sourcing the environment script.

### Backend Changes Are Not Used

Rebuild the backend and then clear stale kernel artifacts:

```sh
scripts/rebuild-triton-riscv.sh
rm -rf "$TRITON_CACHE_DIR"
```

### QEMU Reports `verification failed`

The ELF executed successfully, but a kernel output differed from the corresponding value in `expected`. Check the Python inputs, launch grid, constexpr values, and expected output. The temporary C runner is generated automatically and does not need to be inspected or modified.
