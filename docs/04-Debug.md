# Debugging Triton-RISCV

> **Audience:** first-time users whose vector-add run failed, Kernel Developers,
> and Compiler Contributors.
>
> **Prerequisites:** know whether you are running the Host CPU Quick Start or
> the RISC-V/QEMU tutorial, and keep the exact failing command.
>
> **Outcome:** classify a failure as environment/plugin discovery, frontend,
> lowering, host runtime, numerical correctness, linking, or QEMU.

Debug from the outside in: environment, plugin discovery, compilation stage,
runtime, then numerical correctness. Changing passes before confirming the
active environment is a common way to lose time.

## Vector-Add First Response

Start with the same Host CPU example as the Quick Start:

```sh
python -m pytest python/examples/test_vec_add.py -v -s
```

A healthy launch ends with `PASSED` and prints a small maximum difference. For
an actual numerical assertion, rerun the strict
[Quick Start vector-add check](00-Getting-Started.md#step-7-run-and-strictly-check-host-cpu-vector-add).

Classify what happened before changing code:

| Observation | Likely layer | Next action |
| --- | --- | --- |
| import or missing-tool error before any kernel compilation | environment/plugin discovery | capture paths in the next section |
| Python/Triton rejects syntax, signature, or specialization | frontend | reduce the Python kernel and arguments |
| `tt.mlir` exists but a later IR file does not | lowering | inspect the last good file in [IR inspection](03-IR.md) |
| `ll.ir` exists but Host CPU execution crashes | host runtime or kernel memory access | check launcher arguments, bounds, and sanitizer options |
| Host CPU runs but result differs from PyTorch | kernel semantics or lowering correctness | check grid, mask, strides, aliases, and dtype |
| Host CPU passes but RISC-V link fails | target toolchain/ABI | check triple, `-march`, `-mabi`, and sysroot |
| ELF links but QEMU fails or output verification fails | QEMU/target code/runner data | use the QEMU-specific checks |

The **frontend** turns specialized Python into TTIR. **Lowering** transforms
TTIR through structured/vector MLIR and LLVM IR. The **runtime** loads and
launches a Host CPU object. **QEMU** is a separate emulator used only after a
RISC-V ELF has been generated.

## Capture the Environment First

Run these commands in the same shell that reproduces the failure:

```sh
uname -a
python --version
python -m pip show triton torch

printf 'TRITON_PLUGIN_DIRS=%s\n' "${TRITON_PLUGIN_DIRS:-}"
printf 'LLVM_BINARY_DIR=%s\n' "${LLVM_BINARY_DIR:-}"
printf 'BUDDY_MLIR_BINARY_DIR=%s\n' "${BUDDY_MLIR_BINARY_DIR:-}"
printf 'TRITON_SHARED_OPT_PATH=%s\n' "${TRITON_SHARED_OPT_PATH:-}"
printf 'TRITON_CACHE_DIR=%s\n' "${TRITON_CACHE_DIR:-}"

python - <<'PY'
import triton
from triton.backends.triton_shared import compiler, driver
from triton.backends.triton_shared.paths import _get_triton_shared_opt_path

print("triton:", triton.__file__)
print("compiler:", compiler.__file__)
print("driver:", driver.__file__)
print("triton-shared-opt:", _get_triton_shared_opt_path())
PY
```

This distinguishes a compiler bug from “the shell is using a different Python,
plugin checkout, or toolchain.”

## Confirm Tool Discovery

The backend needs its own optimizer, Buddy tools, and LLVM tools:

```sh
python - <<'PY'
from triton.backends.triton_shared.paths import (
    _get_buddy_opt_path,
    _get_llvm_bin_path,
    _get_triton_shared_opt_path,
)

print("triton-shared-opt:", _get_triton_shared_opt_path())
print("buddy-opt:", _get_buddy_opt_path())
print("mlir-translate:", _get_llvm_bin_path("mlir-translate"))
print("llc:", _get_llvm_bin_path("llc"))
PY
```

Then run lightweight version checks:

```sh
TRITON_SHARED_OPT="$(python -c \
  'from triton.backends.triton_shared.paths import _get_triton_shared_opt_path; print(_get_triton_shared_opt_path())')"
"$TRITON_SHARED_OPT" --version

"$BUDDY_MLIR_BINARY_DIR/buddy-opt" --version
"$LLVM_BINARY_DIR/llc" --version
```

Typical meanings:

- `Unable to locate bundled 'triton-shared-opt'` — the plugin was not built or
  `TRITON_SHARED_OPT_PATH` points to a removed build tree.
- `Unable to locate 'buddy-opt'` — set `BUDDY_MLIR_BINARY_DIR` or source the
  source-build environment helper.
- missing `mlir-translate`/`llc` — `LLVM_BINARY_DIR` is unset or belongs to an
  incomplete LLVM package.
- dialect/attribute parse errors — Triton, Triton-RISCV, Buddy, and LLVM
  revisions may be incompatible.

## Confirm That Your Source Changes Are Installed

Changing `backend/compiler.py`, `backend/driver.py`, or `backend/riscv.py` does
not update an already installed backend automatically. Rebuild:

```sh
scripts/rebuild-triton-riscv.sh
```

Then compare the installed module with the checkout:

```sh
python - <<'PY'
from pathlib import Path
from triton.backends.triton_shared import compiler, driver, riscv

for module in (compiler, driver, riscv):
    print(Path(module.__file__).resolve())
PY
```

`scripts/rebuild-triton-riscv.sh` copies those three files into the installed
plugin and verifies their bytes. If its verification passes but Python prints a
different environment, reactivate the intended venv/conda environment.

After any backend or pass-pipeline change, avoid stale kernel artifacts:

```sh
rm -rf "$TRITON_CACHE_DIR"
```

For one investigation, setting a new cache directory is safer than deleting a
shared cache:

```sh
export TRITON_CACHE_DIR="$PWD/artifacts/cache/debug-case"
```

## Reduce the Failure

Before reading a large IR file, minimize the input:

1. run one pytest node or one Python script;
2. reduce shapes while preserving the failing access pattern;
3. replace the operator with a direct `@triton.jit` kernel;
4. remove autotuning and use one fixed configuration;
5. remove unrelated outputs and intermediate helpers;
6. compare with a small PyTorch reference;
7. keep boundary, non-contiguous, and masked cases if they are part of the bug.

Useful commands:

```sh
python -m pytest python/examples/test_vec_add.py -v -s
python -m pytest path/to/test.py::test_name -v -s
```

`-s` keeps kernel/test prints visible. A minimal reproducer should state the
exact grid, `tl.constexpr` values, dtypes, shapes, strides, and expected result.

## Locate the Failing Compiler Stage

Use a fresh cache and the IR dump workflow:

```sh
export TRITON_SHARED_DUMP_PATH="$PWD/artifacts/ir/debug-case"
export TRITON_CACHE_DIR="$PWD/artifacts/cache/debug-case"
rm -rf "$TRITON_SHARED_DUMP_PATH" "$TRITON_CACHE_DIR"
python -m pytest path/to/test.py::test_name -v -s
```

Interpret the last file written:

| Last file | Likely failing area |
| --- | --- |
| none | frontend did not compile, backend was not selected, or cached code was reused |
| `tt.mlir` | Triton-RISCV structured conversion |
| `ttshared.mlir` | Buddy vector/bufferization pipeline |
| `vector.mlir` | LLVM-dialect lowering or MLIR-to-LLVM translation |
| `ll.mlir` | LLVM translation |
| `ll.ir` | `llc`, linker, launcher, QEMU, or runtime |

See [Inspecting IR](03-IR.md) for stage-specific searches.

If the failure is already present in `ttshared.mlir`, reduce `tt.mlir` to a lit
test and invoke `triton-shared-opt` directly. If `ttshared.mlir` is correct and
Buddy fails, save the exact Buddy command printed by the exception or reproduce
it on the dumped file.

## Debug Incorrect Results

A kernel that compiles can still be wrong. Check these properties in order:

1. **Reference:** Does the PyTorch expression match the intended operator,
   including dtype promotion and in-place behavior?
2. **Launch grid:** Does the grid cover the output exactly? Are program IDs
   interpreted on the intended axes?
3. **Bounds:** Are every load and store guarded when shapes are not block-size
   multiples?
4. **Pointer arithmetic:** Are offsets expressed in elements of the pointee
   type, with correct tensor strides?
5. **Non-contiguity:** Does the test include sliced/transposed tensors if the
   public operator accepts them?
6. **Aliasing:** Can an input and output share storage, and does the lowering
   preserve the required ordering?
7. **Precision:** Are `rtol` and `atol` appropriate for the dtype and reduction
   order without hiding a real error?

Use `torch.testing.assert_close` instead of printing only the maximum error:

```python
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
```

For QEMU, the standalone helper's `expected={...}` mapping performs checks
inside the generated runner. A nonzero exit with `verification failed` means
the ELF ran but at least one output element differed.

## Debug Crashes and Memory Errors

First determine whether the crash is in compilation or execution:

- no `ll.ir` and a subprocess error: compiler pass or tool failure;
- object/ELF created but Python crashes: host launcher or kernel memory access;
- QEMU exits with a signal: RISC-V code, ABI, or generated runner issue;
- link error: target triple, ABI, features, sysroot, or library mismatch.

For RISC-V linker errors, print the detected settings:

```sh
printf '%s\n' "$TRITON_RISCV_CC"
printf '%s\n' "$TRITON_RISCV_SYSROOT"
printf '%s\n' "$TRITON_RISCV_QEMU"
printf '%s\n' "${TRITON_RISCV_MARCH:-}"
printf '%s\n' "${TRITON_RISCV_MABI:-}"
printf '%s\n' "${TRITON_RISCV_LLC_FEATURES:-}"
```

Keep compiler and linker settings aligned. An object generated for one RISC-V
floating-point ABI cannot be linked safely with libraries built for another.

## LLVM Sanitizers and TritonSan

The backend recognizes:

```sh
export TRITON_SHARED_SANITIZER_TYPE=asan   # memory errors
# or
export TRITON_SHARED_SANITIZER_TYPE=tsan   # data races
```

This is an advanced host-CPU workflow. It requires an LLVM build containing the
matching sanitizer runtimes and the project's sanitizer support library;
prebuilt LLVM packages may omit them. Sanitizers are not supported by this
backend on Windows.

The inherited TritonSan tooling and background are documented in
[`triton-san/README.md`](../triton-san/README.md). Treat that guide as a
specialized build path: verify its dependency revisions and paths against the
current Triton-RISCV checkout before using it.

## What to Include in a Bug Report

Provide enough evidence for someone else to reproduce the same layer:

- host OS and architecture;
- Python, Triton, PyTorch, Buddy, and LLVM versions or commits;
- Triton-RISCV commit;
- exact build path (prebuilt or source) and command;
- exact test command and complete error;
- relevant environment variables from [Capture the Environment First](#capture-the-environment-first);
- minimal kernel, inputs, grid, and expected result;
- the last good IR and first bad/failing stage;
- whether a fresh cache changes the result;
- for RISC-V, ELF header/disassembly and target settings.

[Previous: Inspecting IR](03-IR.md) · [Documentation index](README.md) · [Next: Operator migration](05-Operator-Migration.md)
